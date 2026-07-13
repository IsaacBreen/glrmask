use super::accumulator::TerminalsDisallowed;
use super::analysis::EOF;
use super::table::{
    Action,
    AdmissionPolicy,
    GLRTable,
    GuardedShiftCellIndex,
    GuardedStackShift,
    StackShift,
    StackShiftGuard,
};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::{LeveledGSS, Merge, VirtualStack};
use crate::grammar::flat::TerminalID;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;

mod profile;

pub use profile::{
    AdvanceTrace,
    AdvanceTraceGoto,
    AdvanceProfile,
    AdvanceTraceReduce,
    AdvanceTraceStep,
    AdvanceTraceWave,
};

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;
type ReduceBranches = SmallVec<[(ParserGSS, u32, bool); 4]>;
type FloorCrossShift = (u32, u32, bool);

const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 64;
const GUARDED_STACK_TO_STACKS_MAX_DEPTH: usize = 64;
const SMALL_REDUCE_FANOUT_COLLAPSE_MAX_BRANCHES: usize = 8;
const ADVANCE_ASSERT_DISTRIBUTIVITY: u8 = 1 << 0;
const ADVANCE_ASSERT_FAST_PATH_EQUIVALENCE: u8 = 1 << 1;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn assert_row_presence_exact_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("GLRMASK_ASSERT_ROW_PRESENCE_EXACT"))
}

fn advance_assertion_flags() -> u8 {
    static FLAGS: OnceLock<u8> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        let mut flags = 0;
        if env_flag_enabled("GLRMASK_ASSERT_ADVANCE_DISTRIBUTIVITY") {
            flags |= ADVANCE_ASSERT_DISTRIBUTIVITY;
        }
        if env_flag_enabled("GLRMASK_ASSERT_ADVANCE_FAST_PATH_EQUIVALENCE") {
            flags |= ADVANCE_ASSERT_FAST_PATH_EQUIVALENCE;
        }
        flags
    })
}

fn guarded_stack_to_stacks_fallback_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| env_flag_enabled("GLRMASK_DISABLE_GUARDED_STACK_TO_STACKS_FALLBACK"))
}

fn stack_effect_to_stacks_fallback_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| env_flag_enabled("GLRMASK_DISABLE_STACK_EFFECT_TO_STACKS_FALLBACK"))
}

fn advance_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("GLRMASK_PROFILE_ADVANCE_TRACE"))
}

fn trace_action_kind(action: Option<&Action>) -> &'static str {
    match action {
        Some(Action::Shift(..)) => "shift",
        Some(Action::StackShifts(..)) => "stack-shifts",
        Some(Action::GuardedStackShifts(..)) => "guarded-stack-shifts",
        Some(Action::Reduce(..)) => "reduce",
        Some(Action::Split { accept: true, .. }) => "split-accept",
        Some(Action::Split { .. }) => "split",
        Some(Action::Accept) => "accept",
        None => "none",
    }
}

fn trace_reduce_summary(
    table: &GLRTable,
    gss: &ParserGSS,
    lhs_nt: u32,
    pop_len: usize,
) -> AdvanceTraceReduce {
    let mut goto_sources = Vec::new();
    let mut goto_targets = Vec::new();
    for (goto_from, _) in reduce_sources_from_isolated(gss, pop_len) {
        goto_sources.push(goto_from);
        if let Some((target_state, replace)) = table.goto_target(goto_from, lhs_nt) {
            goto_targets.push(AdvanceTraceGoto {
                source_state: goto_from,
                target_state,
                replace,
            });
        }
    }
    goto_sources.sort_unstable();
    goto_sources.dedup();
    goto_targets.sort_by_key(|entry| (entry.source_state, entry.target_state, entry.replace));
    goto_targets.dedup_by(|left, right| {
        left.source_state == right.source_state
            && left.target_state == right.target_state
            && left.replace == right.replace
    });
    AdvanceTraceReduce {
        lhs_nt,
        lhs_name: table.nonterminal_display_name(lhs_nt).map(str::to_owned),
        pop_len: pop_len as u32,
        goto_sources,
        goto_targets,
    }
}

fn trace_action_summary(
    table: &GLRTable,
    source_state: u32,
    gss: &ParserGSS,
    action: Option<&Action>,
) -> AdvanceTraceStep {
    match action {
        Some(Action::Shift(target, replace)) => AdvanceTraceStep {
            source_state,
            action_kind: trace_action_kind(action).to_string(),
            shift_target: Some(*target),
            shift_replace: Some(*replace),
            reduces: Vec::new(),
        },
        Some(Action::StackShifts(..)) | Some(Action::GuardedStackShifts(..)) | Some(Action::Accept) | None => {
            AdvanceTraceStep {
                source_state,
                action_kind: trace_action_kind(action).to_string(),
                shift_target: None,
                shift_replace: None,
                reduces: Vec::new(),
            }
        }
        Some(Action::Reduce(lhs_nt, pop_len)) => AdvanceTraceStep {
            source_state,
            action_kind: trace_action_kind(action).to_string(),
            shift_target: None,
            shift_replace: None,
            reduces: vec![trace_reduce_summary(table, gss, *lhs_nt, *pop_len as usize)],
        },
        Some(Action::Split { shift, reduces, accept }) => AdvanceTraceStep {
            source_state,
            action_kind: if *accept { "split-accept" } else { "split" }.to_string(),
            shift_target: shift.map(|(target, _)| target),
            shift_replace: shift.map(|(_, replace)| replace),
            reduces: reduces
                .iter()
                .map(|&(lhs_nt, pop_len)| trace_reduce_summary(table, gss, lhs_nt, pop_len as usize))
                .collect(),
        },
    }
}

enum AdvancedBranch {
    Stack(VirtualStack<u32, TerminalsDisallowed>),
    Gss(ParserGSS),
}

impl AdvancedBranch {
    fn into_gss(self) -> ParserGSS {
        match self {
            AdvancedBranch::Stack(stack) => stack.into_gss(),
            AdvancedBranch::Gss(gss) => gss,
        }
    }
}

pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {
    let advanced = advance_stacks_core(table, stack.clone(), token);
    assert_advance_oracles(
        advance_assertion_flags(),
        table,
        stack,
        token,
        &advanced,
    );
    advanced
}

/// Like `advance_stacks` but takes ownership of the GSS, avoiding an
/// unnecessary Arc clone when the caller doesn't need the original.
pub(crate) fn advance_stacks_owned(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {
    let assertion_flags = advance_assertion_flags();
    if assertion_flags != 0 {
        let before = stack.clone();
        let advanced = advance_stacks_core(table, stack, token);
        assert_advance_oracles(assertion_flags, table, &before, token, &advanced);
        advanced
    } else {
        advance_stacks_core(table, stack, token)
    }
}

fn normalized_concrete_stacks(
    gss: &ParserGSS,
) -> Vec<(Vec<u32>, TerminalsDisallowed)> {
    let mut normalized: Vec<(Vec<u32>, TerminalsDisallowed)> = Vec::new();
    for (stack, acc) in gss.to_stacks() {
        if let Some((_, existing_acc)) = normalized
            .iter_mut()
            .find(|(existing_stack, _)| *existing_stack == stack)
        {
            *existing_acc = existing_acc.merge(&acc);
        } else {
            normalized.push((stack, acc));
        }
    }
    normalized.sort_by(|left, right| left.0.cmp(&right.0));
    normalized
}

fn merge_concrete_path_accumulator(
    paths: &mut BTreeMap<Vec<u32>, TerminalsDisallowed>,
    stack: Vec<u32>,
    acc: TerminalsDisallowed,
) -> bool {
    if let Some(existing) = paths.get_mut(&stack) {
        let merged = existing.merge(&acc);
        if *existing == merged {
            return false;
        }
        *existing = merged;
        true
    } else {
        paths.insert(stack, acc);
        true
    }
}

fn apply_concrete_stack_effect(
    stack: &[u32],
    pop: usize,
    pushes: &[u32],
) -> Option<Vec<u32>> {
    // An empty concrete parser stack is still a live GSS path. Stack effects
    // are atomic and may therefore pop the final state before pushing a new
    // sequence. Only an actual underflow kills the effect.
    if pop > stack.len() {
        return None;
    }
    let mut next = stack[..stack.len() - pop].to_vec();
    next.extend_from_slice(pushes);
    Some(next)
}

fn concrete_stack_satisfies_guards(stack: &[u32], guards: &[StackShiftGuard]) -> bool {
    guards.iter().all(|guard| {
        let pop = guard.pop as usize;
        pop < stack.len()
            && guard
                .states
                .binary_search(&stack[stack.len() - 1 - pop])
                .is_ok()
    })
}

fn enqueue_concrete_reduction(
    table: &GLRTable,
    stack: &[u32],
    acc: &TerminalsDisallowed,
    nt: u32,
    rhs_len: usize,
    closure: &mut BTreeMap<Vec<u32>, TerminalsDisallowed>,
    queue: &mut VecDeque<Vec<u32>>,
) {
    if rhs_len >= stack.len() {
        return;
    }
    let mut base = stack[..stack.len() - rhs_len].to_vec();
    let goto_from = *base.last().expect("reduction preserves a parser state");
    let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
        return;
    };
    if is_replace {
        let Some(next) = apply_concrete_stack_effect(&base, 1, &[target]) else {
            return;
        };
        base = next;
    } else {
        base.push(target);
    }
    if merge_concrete_path_accumulator(closure, base.clone(), acc.clone()) {
        queue.push_back(base);
    }
}

fn advance_concrete_stacks_reference(
    table: &GLRTable,
    before: &ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    let mut closure = BTreeMap::<Vec<u32>, TerminalsDisallowed>::new();
    let mut queue = VecDeque::<Vec<u32>>::new();
    for (stack, acc) in before.to_stacks() {
        if merge_concrete_path_accumulator(&mut closure, stack.clone(), acc) {
            queue.push_back(stack);
        }
    }

    let mut shifted = BTreeMap::<Vec<u32>, TerminalsDisallowed>::new();
    while let Some(stack) = queue.pop_front() {
        let Some(acc) = closure.get(&stack).cloned() else {
            continue;
        };
        let Some(&state) = stack.last() else {
            continue;
        };
        let Some(action) = table.action(state, token) else {
            continue;
        };

        match action {
            Action::Shift(target, is_replace) => {
                let pop = usize::from(*is_replace);
                if let Some(next) = apply_concrete_stack_effect(&stack, pop, &[*target]) {
                    merge_concrete_path_accumulator(&mut shifted, next, acc);
                }
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    if let Some(next) = apply_concrete_stack_effect(
                        &stack,
                        shift.pop as usize,
                        &shift.pushes,
                    ) {
                        merge_concrete_path_accumulator(&mut shifted, next, acc.clone());
                    }
                }
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    if concrete_stack_satisfies_guards(&stack, &shift.guards)
                        && let Some(next) = apply_concrete_stack_effect(
                            &stack,
                            shift.pop as usize,
                            &shift.pushes,
                        )
                    {
                        merge_concrete_path_accumulator(&mut shifted, next, acc.clone());
                    }
                }
            }
            Action::Reduce(nt, rhs_len) => enqueue_concrete_reduction(
                table,
                &stack,
                &acc,
                *nt,
                *rhs_len as usize,
                &mut closure,
                &mut queue,
            ),
            Action::Split {
                shift,
                reduces,
                accept: _,
            } => {
                if let Some((target, is_replace)) = shift {
                    let is_replace = *is_replace
                        && !table.forwarded_shifts.contains(&(state, token));
                    if let Some(next) = apply_concrete_stack_effect(
                        &stack,
                        usize::from(is_replace),
                        &[*target],
                    ) {
                        merge_concrete_path_accumulator(&mut shifted, next, acc.clone());
                    }
                }
                for &(nt, rhs_len) in reduces {
                    enqueue_concrete_reduction(
                        table,
                        &stack,
                        &acc,
                        nt,
                        rhs_len as usize,
                        &mut closure,
                        &mut queue,
                    );
                }
            }
            Action::Accept => {}
        }
    }

    if shifted.is_empty() {
        ParserGSS::empty()
    } else {
        let stacks = shifted.into_iter().collect::<Vec<_>>();
        ParserGSS::from_stacks(&stacks)
    }
}

#[cold]
fn assert_advance_matches_concrete_reference(
    table: &GLRTable,
    before: &ParserGSS,
    token: TerminalID,
    actual: &ParserGSS,
) {
    let expected = advance_concrete_stacks_reference(table, before, token);
    assert_eq!(
        normalized_concrete_stacks(actual),
        normalized_concrete_stacks(&expected),
        "parser fast-path advance mismatch for terminal {token}: construction={:?} before={:?}",
        table.construction,
        normalized_concrete_stacks(before),
    );
}

#[cold]
fn assert_advance_distributes_over_concrete_paths(
    table: &GLRTable,
    before: &ParserGSS,
    token: TerminalID,
    actual: &ParserGSS,
) {
    let concrete_paths = before.to_stacks();
    if concrete_paths.len() <= 1 {
        return;
    }

    let mut expected = ParserGSS::empty();
    for (stack, acc) in concrete_paths {
        merge_into(
            &mut expected,
            advance_stacks_core(table, ParserGSS::from_single_stack(stack, acc), token),
        );
    }

    assert_eq!(
        normalized_concrete_stacks(actual),
        normalized_concrete_stacks(&expected),
        "parser advance is not distributive over GSS path union for terminal {token}: construction={:?} before={:?}",
        table.construction,
        normalized_concrete_stacks(before),
    );
}

#[inline]
fn assert_advance_oracles(
    assertion_flags: u8,
    table: &GLRTable,
    before: &ParserGSS,
    token: TerminalID,
    actual: &ParserGSS,
) {
    if assertion_flags & ADVANCE_ASSERT_FAST_PATH_EQUIVALENCE != 0 {
        assert_advance_matches_concrete_reference(table, before, token, actual);
    }
    if assertion_flags & ADVANCE_ASSERT_DISTRIBUTIVITY != 0 {
        assert_advance_distributes_over_concrete_paths(table, before, token, actual);
    }
}

pub(crate) fn advance_stacks_profiled(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> (ParserGSS, AdvanceProfile) {
    let (advanced, profile) = advance_stacks_profiled_core(table, stack, token);
    assert_advance_oracles(
        advance_assertion_flags(),
        table,
        stack,
        token,
        &advanced,
    );
    (advanced, profile)
}

fn advance_stacks_profiled_core(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> (ParserGSS, AdvanceProfile) {
    use std::time::Instant;

    let total_start = Instant::now();
    let clone_start = Instant::now();
    let mut gss = stack.clone();
    let mut profile = AdvanceProfile {
        clone_ns: clone_start.elapsed().as_nanos() as u64,
        top_states: stack.peek_values().len() as u32,
        gss_depth: stack.max_depth(),
        vstack_len: stack.try_virtual_stack().map_or(0, |vstack| vstack.len() as u32),
        trace: advance_trace_enabled().then(AdvanceTrace::default),
        ..AdvanceProfile::default()
    };

    let fast_path_start = Instant::now();
    if let Some(state) = gss.single_exclusive_top_value() {
        match table.action(state, token) {
            Some(Action::Shift(target, is_replace)) => {
                if let Some(trace) = profile.trace.as_mut() {
                    trace.det_steps.push(trace_action_summary(
                        table,
                        state,
                        &gss,
                        Some(&Action::Shift(*target, *is_replace)),
                    ));
                }
                profile.pure_shift = true;
                profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                let apply_start = Instant::now();
                let shifted = if *is_replace {
                    gss.popn(1).push(*target)
                } else {
                    gss.push(*target)
                };
                profile.stack_shift_apply_ns = apply_start.elapsed().as_nanos() as u64;
                profile.total_ns = total_start.elapsed().as_nanos() as u64;
                return (shifted, profile);
            }
            Some(Action::StackShifts(shifts)) => {
                if let Some(trace) = profile.trace.as_mut() {
                    trace.det_steps.push(trace_action_summary(table, state, &gss, Some(&Action::StackShifts(shifts.clone()))));
                }
                profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                let apply_start = Instant::now();
                let shifted = apply_stack_shifts(gss, shifts);
                profile.stack_shift_apply_ns = apply_start.elapsed().as_nanos() as u64;
                profile.total_ns = total_start.elapsed().as_nanos() as u64;
                return (shifted, profile);
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                if let Some(trace) = profile.trace.as_mut() {
                    trace.det_steps.push(trace_action_summary(
                        table,
                        state,
                        &gss,
                        Some(&Action::GuardedStackShifts(shifts.clone())),
                    ));
                }
                profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                let apply_start = Instant::now();
                let shifted = apply_guarded_stack_shifts(
                    gss,
                    shifts,
                    table.guarded_shift_index(state, token),
                );
                profile.stack_shift_apply_ns = apply_start.elapsed().as_nanos() as u64;
                profile.total_ns = total_start.elapsed().as_nanos() as u64;
                return (shifted, profile);
            }
            Some(Action::Reduce(nt, len)) => {
                if let Some(trace) = profile.trace.as_mut() {
                    trace.det_steps.push(trace_action_summary(
                        table,
                        state,
                        &gss,
                        Some(&Action::Reduce(*nt, *len)),
                    ));
                }
                if let Some(shifted) = try_collapse_small_reduce_fanout(table, &gss, token) {
                    profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;
                    profile.stack_shift_apply_ns = profile.fast_path_ns;
                    profile.total_ns = total_start.elapsed().as_nanos() as u64;
                    return (shifted, profile);
                }
            }
            _ => {}
        }
    }
    profile.fast_path_ns = fast_path_start.elapsed().as_nanos() as u64;

    let frontier_start = Instant::now();
    if let Some(shifted) = advance_pure_frontier_shifts(table, &gss, token) {
        profile.pure_shift = true;
        profile.stack_shift_apply_ns = frontier_start.elapsed().as_nanos() as u64;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (shifted, profile);
    }

    let det_start = Instant::now();
    let det_ok = advance_deterministically_profiled(table, &mut gss, token, &mut profile);
    profile.det_ns = det_start.elapsed().as_nanos() as u64;
    if det_ok {
        profile.deterministic_finished = true;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (gss, profile);
    }

    let post_det_frontier_start = Instant::now();
    if let Some(shifted) = advance_pure_frontier_shifts(table, &gss, token) {
        profile.pure_shift = true;
        profile.stack_shift_apply_ns += post_det_frontier_start.elapsed().as_nanos() as u64;
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return (shifted, profile);
    }

    let nondet_start = Instant::now();
    profile.nondeterministic_entered = true;
    let gss = advance_nondeterministically_profiled(table, gss, token, &mut profile);
    profile.nondet_ns = nondet_start.elapsed().as_nanos() as u64;
    profile.total_ns = total_start.elapsed().as_nanos() as u64;
    (gss, profile)
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
    if let Some(state) = gss.single_exclusive_top_value() {
        if let Some(Action::Shift(target, is_replace)) = table.action(state, token) {
            return if *is_replace {
                gss.popn(1).push(*target)
            } else {
                gss.push(*target)
            };
        }
        if let Some(Action::StackShifts(shifts)) = table.action(state, token) {
            return apply_stack_shifts(gss, shifts);
        }
        if let Some(Action::GuardedStackShifts(shifts)) = table.action(state, token) {
            return apply_guarded_stack_shifts(gss, shifts, table.guarded_shift_index(state, token));
        }
        if matches!(table.action(state, token), Some(Action::Reduce(..)))
            && let Some(shifted) = try_collapse_small_reduce_fanout(table, &gss, token)
        {
            return shifted;
        }
    }

    if let Some(shifted) = advance_pure_frontier_shifts(table, &gss, token) {
        return shifted;
    }

    if advance_deterministically(table, &mut gss, token) {
        return gss;
    }

    if let Some(shifted) = advance_pure_frontier_shifts(table, &gss, token) {
        return shifted;
    }

    advance_nondeterministically(table, gss, token)
}

fn try_collapse_small_reduce_fanout(
    table: &GLRTable,
    gss: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    // This optimization only applies when reducing exposes multiple isolated
    // branches. A virtual stack is a single concrete path, so the existing
    // branch construction would produce at most one branch and return `None`.
    if gss.try_virtual_stack().is_some() {
        return None;
    }

    let state = gss.single_exclusive_top_value()?;
    let Action::Reduce(nt, len) = table.action(state, token)? else {
        return None;
    };

    let branches = reduce_branches_from_isolated(table, gss, *nt, *len as usize);
    if branches.len() <= 1 || branches.len() > SMALL_REDUCE_FANOUT_COLLAPSE_MAX_BRANCHES {
        return None;
    }

    let mut collapsed: Option<ParserGSS> = None;
    for (base, target, is_replace) in branches {
        let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
        if !det_ok {
            return None;
        }

        let branch = branch.into_gss();
        if branch.is_empty() {
            return None;
        }

        if let Some(existing) = collapsed.as_ref() {
            if &branch != existing {
                return None;
            }
        } else {
            collapsed = Some(branch);
        }
    }

    collapsed
}

#[inline]
fn pure_frontier_shift(action: &Action) -> Option<(u32, bool)> {
    match action {
        Action::Shift(target, is_replace) => Some((*target, *is_replace)),
        Action::StackShifts(shifts)
            if shifts.len() == 1 && shifts[0].pushes.len() == 1 && shifts[0].pop <= 1 =>
        {
            Some((shifts[0].pushes[0], shifts[0].pop == 1))
        }
        _ => None,
    }
}

fn effective_pure_frontier_shift(
    table: &GLRTable,
    state: u32,
    token: TerminalID,
    action: &Action,
) -> Option<(u32, bool)> {
    match action {
        Action::Shift(target, is_replace) => Some((
            *target,
            *is_replace && !table.forwarded_shifts.contains(&(state, token)),
        )),
        Action::StackShifts(shifts)
            if shifts.len() == 1 && shifts[0].pushes.len() == 1 && shifts[0].pop <= 1 =>
        {
            Some((shifts[0].pushes[0], shifts[0].pop == 1))
        }
        _ => None,
    }
}

fn advance_pure_frontier_shifts(
    table: &GLRTable,
    gss: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = gss.peek_values();
    if states.len() <= 1 {
        return None;
    }

    let mut shifts: SmallVec<[(u32, u32, bool); 8]> = SmallVec::new();
    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        let (target, replace_top) = pure_frontier_shift(action)?;
        shifts.push((state, target, replace_top));
    }
    if shifts.is_empty() {
        return None;
    }
    if let Some(shifted) = gss.try_apply_selective_top_pure_shifts(shifts.iter().copied()) {
        return Some(shifted);
    }
    Some(gss.apply_top_pure_shifts(shifts))
}

fn try_advance_single_alt_pop1_common_suffix_stackshift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let mut grouped_prefixes: SmallVec<[(u32, SmallVec<[(u32, u32, bool); 8]>); 4]> = SmallVec::new();
    let mut pure_shifts: SmallVec<[(u32, u32, bool); 4]> = SmallVec::new();
    let mut saw_stack_shift = false;

    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        match action {
            Action::StackShifts(shifts) => {
                let [shift] = shifts.as_slice() else {
                    return None;
                };
                if shift.pop != 1 || shift.pushes.len() != 2 {
                    return None;
                }
                saw_stack_shift = true;
                let prefix = shift.pushes[0];
                let suffix = shift.pushes[1];
                if let Some((_, prefixes)) = grouped_prefixes
                    .iter_mut()
                    .find(|(existing_suffix, _)| *existing_suffix == suffix)
                {
                    prefixes.push((state, prefix, true));
                } else {
                    let mut prefixes = SmallVec::new();
                    prefixes.push((state, prefix, true));
                    grouped_prefixes.push((suffix, prefixes));
                }
            }
            Action::Shift(target, is_replace) => {
                pure_shifts.push((
                    state,
                    *target,
                    *is_replace && !table.forwarded_shifts.contains(&(state, token)),
                ));
            }
            _ => return None,
        }
    }

    if !saw_stack_shift {
        return None;
    }

    let mut out = ParserGSS::empty();
    if !pure_shifts.is_empty() {
        merge_into(&mut out, closure.apply_top_pure_shifts(pure_shifts));
    }
    for (suffix, prefixes) in grouped_prefixes {
        merge_into(&mut out, closure.apply_top_pure_shifts(prefixes).push(suffix));
    }

    (!out.is_empty()).then_some(out)
}

fn try_advance_single_active_pop1_reduce_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let mut reduction: Option<(u32, u32)> = None;
    for state in states {
        match table.action(state, token) {
            None => {}
            Some(Action::Reduce(nt, 1)) => {
                if reduction.replace((state, *nt)).is_some() {
                    return None;
                }
            }
            Some(Action::Split { shift: None, reduces, accept: false })
                if reduces.len() == 1 && reduces[0].1 == 1 =>
            {
                if reduction.replace((state, reduces[0].0)).is_some() {
                    return None;
                }
            }
            _ => return None,
        }
    }

    let (state, nt) = reduction?;
    let popped = closure.pop_top_value(&state);
    if popped.is_empty() {
        return None;
    }

    if let Some(mut stack) = popped.try_virtual_stack() {
        let Some(&goto_from) = stack.top() else {
            return None;
        };
        let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
            return None;
        };
        if is_replace {
            stack.replace_top(target);
        } else {
            stack.push(target);
        }
        let (branch, det_ok) = advance_deterministically_from_vstack_raw(table, stack, token);
        return det_ok.then(|| branch.into_gss());
    }

    let goto_sources = popped.peek_values();
    let [goto_from] = goto_sources.as_slice() else {
        return None;
    };
    let (target, is_replace) = table.goto_target(*goto_from, nt)?;
    let (branch, det_ok) = advance_reduce_branch(table, popped, target, is_replace, token);
    det_ok.then(|| branch.into_gss())
}

fn try_advance_pop1_common_base_reduce_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let base = closure.pop1_common_interface_base()?;
    let base_stack = base.try_virtual_stack()?;
    let Some(&goto_from) = base_stack.top() else {
        return None;
    };

    let mut targets = Vec::with_capacity(states.len());
    let mut fallback = ParserGSS::empty();
    let mut saw_reduce = false;

    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        let nt = match action {
            Action::Reduce(nt, 1) => *nt,
            Action::Split {
                shift: None,
                reduces,
                accept: false,
            } if reduces.len() == 1 && reduces[0].1 == 1 => reduces[0].0,
            _ => return None,
        };
        saw_reduce = true;

        let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
            continue;
        };

        if !is_replace
            && let Some(next_action) = table.action(target, token)
            && let Some((next_target, true)) =
                effective_pure_frontier_shift(table, target, token, next_action)
        {
            targets.push(next_target);
            continue;
        }

        let mut stack = base_stack.clone();
        if is_replace {
            if !stack.replace_top(target) {
                return None;
            }
        } else {
            stack.push(target);
        }

        let (branch, det_ok) = advance_deterministically_from_vstack_raw(table, stack, token);
        if !det_ok {
            return None;
        }
        match branch {
            AdvancedBranch::Stack(stack) => {
                if let Some(top) = stack.single_top_extension_of(&base_stack) {
                    targets.push(top);
                } else {
                    merge_into(&mut fallback, stack.into_gss());
                }
            }
            AdvancedBranch::Gss(branch) => merge_into(&mut fallback, branch),
        }
    }

    if !saw_reduce {
        return None;
    }

    let mut out = ParserGSS::empty();
    if !targets.is_empty() {
        if let Some(grouped) = base_stack
            .clone()
            .into_gss_after_popping_and_pushing_single_branches(0, targets.iter())
        {
            merge_into(&mut out, grouped);
        }
    }
    merge_into(&mut out, fallback);
    (!out.is_empty()).then_some(out)
}

fn try_advance_mixed_top_pop1_shift_reduce_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let mut closure = closure.clone();
    let mut shifted = ParserGSS::empty();

    let mut seen_frontiers: FxHashSet<Vec<u32>> = FxHashSet::default();
    loop {
        let states = closure.peek_values();
        if states.len() < 2 {
            return None;
        }
        let mut frontier_signature = states.to_vec();
        frontier_signature.sort_unstable();
        if !seen_frontiers.insert(frontier_signature) {
            return None;
        }

        let mut pure_shifts: SmallVec<[(u32, u32, bool); 16]> = SmallVec::new();
        let mut stack_shifts: SmallVec<[(u32, &[StackShift]); 16]> = SmallVec::new();
        let mut reductions: SmallVec<[(u32, u32); 16]> = SmallVec::new();

        for state in states {
            let Some(action) = table.action(state, token) else {
                continue;
            };
            match action {
                Action::Shift(target, is_replace) => {
                    pure_shifts.push((state, *target, *is_replace));
                }
                Action::StackShifts(shifts) => {
                    if let Some((target, is_replace)) = pure_frontier_shift(action) {
                        pure_shifts.push((state, target, is_replace));
                    } else {
                        stack_shifts.push((state, shifts));
                    }
                }
                Action::Reduce(nt, 1) => {
                    reductions.push((state, *nt));
                }
                Action::Split { shift, reduces, accept: false }
                    if reduces.iter().all(|(_, len)| *len == 1) =>
                {
                    if let Some((target, is_replace)) = shift {
                        pure_shifts.push((
                            state,
                            *target,
                            *is_replace && !table.forwarded_shifts.contains(&(state, token)),
                        ));
                    }
                    for &(nt, _) in reduces {
                        reductions.push((state, nt));
                    }
                }
                _ => return None,
            }
        }

        if pure_shifts.is_empty() && stack_shifts.is_empty() && reductions.is_empty() {
            return None;
        }
        if reductions.is_empty() && !stack_shifts.is_empty() {
            return None;
        }

        if !pure_shifts.is_empty() {
            let shifted_wave = match pure_shifts.as_slice() {
                [shift] => closure
                    .try_apply_selective_top_pure_shifts([*shift])
                    .unwrap_or_else(|| closure.apply_top_pure_shifts([*shift])),
                _ => closure.apply_top_pure_shifts(pure_shifts),
            };
            merge_into(&mut shifted, shifted_wave);
        }

        let common_pop1_base = closure.pop1_common_interface_base();

        for (state, shifts) in stack_shifts {
            let branch_base = common_pop1_base
                .as_ref()
                .map(|base| base.clone().push(state))
                .unwrap_or_else(|| closure.isolate(Some(state)));
            merge_into(&mut shifted, apply_stack_shifts(branch_base, shifts));
        }

        if reductions.is_empty() {
            return (!shifted.is_empty()).then_some(shifted);
        }

        let common_reduce_source = common_pop1_base.as_ref().and_then(|base| {
            let values = base.peek_values();
            let [source] = values.as_slice() else {
                return None;
            };
            Some((base.clone(), *source))
        });

        let mut next = ParserGSS::empty();
        for (state, nt) in reductions {
            if let Some((base, goto_from)) = common_reduce_source.as_ref() {
                let Some((target, is_replace)) = table.goto_target(*goto_from, nt) else { continue; };
                let (branch, det_ok) = advance_reduce_branch(table, base.clone(), target, is_replace, token);
                if det_ok {
                    match branch {
                        AdvancedBranch::Stack(stack) => {
                            shifted = shifted.absorb_vstack_same_acc_owned(stack);
                        }
                        AdvancedBranch::Gss(branch) => {
                            merge_into(&mut shifted, branch);
                        }
                    }
                } else {
                    merge_into(&mut next, branch.into_gss());
                }
                continue;
            }

            let popped = closure.pop_top_value(&state);
            if popped.is_empty() {
                continue;
            }

            if let Some(mut stack) = popped.try_virtual_stack() {
                let Some(&goto_from) = stack.top() else { continue; };
                let Some((target, is_replace)) = table.goto_target(goto_from, nt) else { continue; };
                if is_replace {
                    stack.replace_top(target);
                } else {
                    stack.push(target);
                }
                let (branch, det_ok) = advance_deterministically_from_vstack_raw(table, stack, token);
                if det_ok {
                    match branch {
                        AdvancedBranch::Stack(stack) => {
                            shifted = shifted.absorb_vstack_same_acc_owned(stack);
                        }
                        AdvancedBranch::Gss(branch) => {
                            merge_into(&mut shifted, branch);
                        }
                    }
                } else {
                    merge_into(&mut next, branch.into_gss());
                }
                continue;
            }

            let goto_sources = popped.peek_values();
            for goto_from in goto_sources.iter().copied() {
                let Some((target, is_replace)) = table.goto_target(goto_from, nt) else { continue; };
                let base = if goto_sources.len() == 1 {
                    popped.clone()
                } else {
                    popped.isolate(Some(goto_from))
                };
                let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
                if det_ok {
                    match branch {
                        AdvancedBranch::Stack(stack) => {
                            shifted = shifted.absorb_vstack_same_acc_owned(stack);
                        }
                        AdvancedBranch::Gss(branch) => {
                            merge_into(&mut shifted, branch);
                        }
                    }
                } else {
                    merge_into(&mut next, branch.into_gss());
                }
            }
        }

        if next.is_empty() {
            return (!shifted.is_empty()).then_some(shifted);
        }
        closure = next;
    }
}

fn try_advance_pop1_stackshift_shift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    if let Some(shifted) = try_advance_single_alt_pop1_common_suffix_stackshift_wave(table, closure, token) {
        return Some(shifted);
    }

    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }
    let base = closure.pop1_common_interface_base()?;

    let mut out = ParserGSS::empty();
    let mut saw_stack_shift = false;
    let mut saw_action = false;
    for state in states {
        let Some(action) = table.action(state, token) else {
            continue;
        };
        saw_action = true;
        match action {
            Action::StackShifts(shifts) => {
                if let Some((target, is_replace)) = pure_frontier_shift(action) {
                    let branch = if is_replace {
                        base.clone().push(target)
                    } else {
                        base.clone().push(state).push(target)
                    };
                    merge_into(&mut out, branch);
                } else {
                    saw_stack_shift = true;
                    merge_into(&mut out, apply_stack_shifts(base.clone().push(state), shifts));
                }
            }
            Action::Shift(target, is_replace) => {
                let is_replace = *is_replace && !table.forwarded_shifts.contains(&(state, token));
                let branch = if is_replace {
                    base.clone().push(*target)
                } else {
                    base.clone().push(state).push(*target)
                };
                merge_into(&mut out, branch);
            }
            _ => return None,
        }
    }

    (saw_action && saw_stack_shift && !out.is_empty()).then_some(out)
}

fn try_advance_pop1_reduce_plus_stackshift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if states.len() < 2 {
        return None;
    }

    let base = if let Some(base) = closure.pop1_common_interface_base() {
        base
    } else {
        let base = closure.popn(1);
        let mut reconstructed = ParserGSS::empty();
        for &state in &states {
            merge_into(&mut reconstructed, base.clone().push(state));
        }
        if &reconstructed != closure {
            return None;
        }
        base
    };
    let base_values = base.peek_values();
    let [base_top] = base_values.as_slice() else {
        return None;
    };
    let base_top = *base_top;

    let mut reduce_nt: Option<u32> = None;
    let mut shifted = ParserGSS::empty();
    for state in states {
        let action = table.action(state, token)?;
        match action {
            Action::Reduce(nt, 1) => {
                if reduce_nt.replace(*nt).is_some() {
                    return None;
                }
            }
            Action::StackShifts(shifts) => {
                merge_into(&mut shifted, apply_stack_shifts(base.push(state), shifts));
            }
            Action::Shift(target, is_replace) => {
                let branch = base.push(state);
                let is_replace = *is_replace && !table.forwarded_shifts.contains(&(state, token));
                if is_replace {
                    merge_into(&mut shifted, branch.popn(1).push(*target));
                } else {
                    merge_into(&mut shifted, branch.push(*target));
                }
            }
            _ => return None,
        }
    }

    let reduce_nt = reduce_nt?;
    if shifted.is_empty() {
        return None;
    }

    let (target, is_replace) = table.goto_target(base_top, reduce_nt)?;
    let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
    if !det_ok {
        return None;
    }
    merge_into(&mut shifted, branch.into_gss());
    Some(shifted)
}

fn rebuild_floor_cross_from_shifts(
    popped: ParserGSS,
    shifts: SmallVec<[FloorCrossShift; 8]>,
) -> ParserGSS {
    if shifts.iter().any(|(_, _, is_replace)| *is_replace) {
        popped.apply_top_pure_shifts(shifts)
    } else {
        popped.remap_top_values_owned(
            shifts
                .into_iter()
                .map(|(goto_from, target, _)| (goto_from, target)),
        )
    }
}

fn push_states(mut gss: ParserGSS, states: &[u32]) -> ParserGSS {
    for &state in states {
        gss = gss.push(state);
    }
    gss
}

fn common_stack_shift_suffix_len(pushes: &[&[u32]]) -> usize {
    let Some(first) = pushes.first() else {
        return 0;
    };
    let mut suffix_len = 0;
    'suffix: while suffix_len < first.len() {
        let state = first[first.len() - 1 - suffix_len];
        for pushes in &pushes[1..] {
            if suffix_len >= pushes.len() || pushes[pushes.len() - 1 - suffix_len] != state {
                break 'suffix;
            }
        }
        suffix_len += 1;
    }
    suffix_len
}

fn apply_push_sequences(base: ParserGSS, pushes: &[&[u32]]) -> ParserGSS {
    match pushes {
        [] => ParserGSS::empty(),
        [pushes] => push_states(base, pushes),
        _ => {
            let common_suffix_len = common_stack_shift_suffix_len(pushes);
            if common_suffix_len > 0 {
                let mut prefixes = ParserGSS::empty();
                for pushes in pushes {
                    let prefix_len = pushes.len() - common_suffix_len;
                    merge_into(&mut prefixes, push_states(base.clone(), &pushes[..prefix_len]));
                }
                let suffix = &pushes[0][pushes[0].len() - common_suffix_len..];
                push_states(prefixes, suffix)
            } else {
                let mut out = ParserGSS::empty();
                for pushes in pushes {
                    merge_into(&mut out, push_states(base.clone(), pushes));
                }
                out
            }
        }
    }
}

fn apply_stack_shifts(gss: ParserGSS, shifts: &[StackShift]) -> ParserGSS {
    if let Some(stack) = gss.try_virtual_stack()
        && let Some(first) = shifts.first()
        && !first.pushes.is_empty()
        && shifts
            .iter()
            .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
    {
        if shifts.iter().all(|shift| shift.pushes.len() == 1)
            && let Some(shifted) = stack.clone().into_gss_after_popping_and_pushing_single_branches(
                first.pop as usize,
                shifts.iter().map(|shift| &shift.pushes[0]),
            )
        {
            return shifted;
        }
        if let Some(shifted) = stack.into_gss_after_popping_and_pushing_branches(
            first.pop as usize,
            shifts.iter().map(|shift| shift.pushes.as_slice()),
        ) {
            return shifted;
        }
    }

    if let Some(shifted) = gss.apply_stack_effects_to_single_concrete_path(
        shifts
            .iter()
            .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
        if stack_effect_to_stacks_fallback_disabled() {
            0
        } else {
            SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH
        },
    ) {
        return shifted;
    }
    let mut out = ParserGSS::empty();
    let mut groups: SmallVec<[(u32, SmallVec<[&[u32]; 4]>); 4]> = SmallVec::new();

    for shift in shifts {
        if let Some((_, group_pushes)) = groups.iter_mut().find(|(pop, _)| *pop == shift.pop) {
            group_pushes.push(shift.pushes.as_slice());
        } else {
            let mut group_pushes = SmallVec::new();
            group_pushes.push(shift.pushes.as_slice());
            groups.push((shift.pop, group_pushes));
        }
    }

    for (pop, pushes) in groups {
        let base = gss.popn(pop as isize);
        merge_into(&mut out, apply_push_sequences(base, &pushes));
    }
    out
}

fn guarded_stack_shifts_are_decidable_from_vstack(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    shifts: &[GuardedStackShift],
) -> bool {
    if !stack.has_hidden_floor_values() {
        return true;
    }

    let visible_len = stack.len();
    for shift in shifts {
        let mut ruled_out = false;
        for guard in &shift.guards {
            let pop = guard.pop as usize;
            if pop >= visible_len {
                // This still-live shift needs a branch-specific value from the
                // hidden floor. The virtual prefix cannot decide the guard.
                return false;
            }
            let state = stack
                .top_after_popping(pop)
                .expect("guard depth lies within visible virtual-stack prefix");
            if guard.states.binary_search(state).is_err() {
                ruled_out = true;
                break;
            }
        }

        if !ruled_out && shift.pop as usize > visible_len {
            // The guards are satisfied by the visible prefix, but applying the
            // effect would pop through the branched floor.
            return false;
        }
    }

    true
}

fn apply_guarded_stack_shifts_from_vstack(
    stack: VirtualStack<u32, TerminalsDisallowed>,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> ParserGSS {
    if guarded_stack_shifts_are_decidable_from_vstack(&stack, shifts) {
        apply_guarded_stack_shifts_to_vstack(&stack, shifts, index)
    } else {
        apply_guarded_stack_shifts(stack.into_gss(), shifts, index)
    }
}

pub(crate) fn apply_guarded_stack_shifts_fast(
    gss: &ParserGSS,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> Option<ParserGSS> {
    if let Some(stack) = gss.try_virtual_stack()
        && guarded_stack_shifts_are_decidable_from_vstack(&stack, shifts)
    {
        return Some(apply_guarded_stack_shifts_to_vstack(&stack, shifts, index));
    }

    if !guarded_stack_to_stacks_fallback_disabled()
        && let Some(shifted) = gss.apply_guarded_stack_effects_to_single_concrete_path(
            shifts.iter().map(|shift| {
                (
                    shift
                        .guards
                        .iter()
                        .map(|guard| (guard.pop as usize, guard.states.as_slice())),
                    shift.pop as usize,
                    shift.pushes.as_slice(),
                )
            }),
            GUARDED_STACK_TO_STACKS_MAX_DEPTH,
        )
    {
        return Some(shifted);
    }

    None
}

fn apply_guarded_stack_shifts(
    gss: ParserGSS,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> ParserGSS {
    if let Some(shifted) = apply_guarded_stack_shifts_fast(&gss, shifts, index) {
        return shifted;
    }

    let mut out = ParserGSS::empty();

    for shift in shifts {
        debug_assert!(shift.guards.windows(2).all(|w| w[0].pop <= w[1].pop));
        debug_assert!(shift.guards.iter().all(|guard| guard.pop <= shift.pop));

        let mut base = gss.clone();
        let mut depth = 0u32;
        let mut dead = false;

        for guard in &shift.guards {
            if guard.pop < depth {
                dead = true;
                break;
            }

            base = base.popn((guard.pop - depth) as isize);
            if base.is_empty() {
                dead = true;
                break;
            }

            let mut filtered = ParserGSS::empty();
            for &state in &guard.states {
                merge_into(&mut filtered, base.isolate(Some(state)));
            }

            base = filtered;
            if base.is_empty() {
                dead = true;
                break;
            }

            depth = guard.pop;
        }

        if dead || shift.pop < depth {
            continue;
        }

        let branch = push_states(base.popn((shift.pop - depth) as isize), &shift.pushes);
        merge_into(&mut out, branch);
    }

    out
}

fn indexed_guarded_shift_candidates(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    index: &GuardedShiftCellIndex,
) -> SmallVec<[u32; 8]> {
    let mut counts: FxHashMap<u32, u16> = FxHashMap::default();

    for &pop in &index.guard_pops {
        let Some(state) = stack.top_after_popping(pop as usize).copied() else {
            continue;
        };
        if let Some(shift_indices) = index.by_guard_key.get(&(pop, state)) {
            for &shift_index in shift_indices.iter() {
                *counts.entry(shift_index).or_insert(0) += 1;
            }
        }
    }

    let mut candidates = SmallVec::<[u32; 8]>::new();
    for (shift_index, count) in counts {
        if index
            .guard_counts
            .get(shift_index as usize)
            .is_some_and(|required| *required == count)
        {
            candidates.push(shift_index);
        }
    }
    candidates.extend(index.unguarded_indices.iter().copied());
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn apply_guarded_stack_shifts_to_vstack(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> ParserGSS {
    let mut groups: SmallVec<[(u32, SmallVec<[&[u32]; 4]>); 4]> = SmallVec::new();
    let mut empty_pushes: SmallVec<[u32; 4]> = SmallVec::new();
    let stack_len = stack.len();
    let mut state_after_pop_cache: SmallVec<[(u32, Option<u32>); 8]> = SmallVec::new();

    #[inline]
    fn state_after_popping(
        stack: &VirtualStack<u32, TerminalsDisallowed>,
        cache: &mut SmallVec<[(u32, Option<u32>); 8]>,
        pop: u32,
    ) -> Option<u32> {
        if let Some((_, cached)) = cache.iter().find(|(cached_pop, _)| *cached_pop == pop) {
            return *cached;
        }
        let value = stack.top_after_popping(pop as usize).copied();
        cache.push((pop, value));
        value
    }

    fn consider_guarded_shift<'a>(
        stack: &VirtualStack<u32, TerminalsDisallowed>,
        stack_len: usize,
        state_after_pop_cache: &mut SmallVec<[(u32, Option<u32>); 8]>,
        groups: &mut SmallVec<[(u32, SmallVec<[&'a [u32]; 4]>); 4]>,
        empty_pushes: &mut SmallVec<[u32; 4]>,
        shift: &'a GuardedStackShift,
    ) {
        debug_assert!(shift.guards.windows(2).all(|w| w[0].pop <= w[1].pop));
        debug_assert!(shift.guards.iter().all(|guard| guard.pop <= shift.pop));

        let mut dead = false;
        for guard in &shift.guards {
            let Some(state) = state_after_popping(stack, state_after_pop_cache, guard.pop) else {
                dead = true;
                break;
            };
            if guard.states.binary_search(&state).is_err() {
                dead = true;
                break;
            }
        }

        if dead || shift.pop as usize > stack_len {
            return;
        }

        if shift.pushes.is_empty() {
            empty_pushes.push(shift.pop);
        } else if let Some((_, pushes)) = groups.iter_mut().find(|(pop, _)| *pop == shift.pop) {
            pushes.push(shift.pushes.as_slice());
        } else {
            let mut pushes = SmallVec::new();
            pushes.push(shift.pushes.as_slice());
            groups.push((shift.pop, pushes));
        }
    }

    if let Some(index) = index {
        for shift_index in indexed_guarded_shift_candidates(stack, index) {
            if let Some(shift) = shifts.get(shift_index as usize) {
                consider_guarded_shift(
                    stack,
                    stack_len,
                    &mut state_after_pop_cache,
                    &mut groups,
                    &mut empty_pushes,
                    shift,
                );
            }
        }
    } else {
        for shift in shifts {
            consider_guarded_shift(
                stack,
                stack_len,
                &mut state_after_pop_cache,
                &mut groups,
                &mut empty_pushes,
                shift,
            );
        }
    }

    let mut out = ParserGSS::empty();
    for (pop, pushes) in groups {
        if let Some(branch) =
            stack.clone().into_gss_after_popping_and_pushing_branches(pop as usize, pushes)
        {
            merge_into(&mut out, branch);
        }
    }
    for pop in empty_pushes {
        let mut branch = stack.clone();
        if branch.pop(pop as usize) == 0 {
            merge_into(&mut out, branch.into_gss());
        }
    }
    out
}

#[inline]
fn virtual_stack_satisfies_guards(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    guards: &[StackShiftGuard],
) -> bool {
    let mut cursor = stack.clone();
    let mut depth = 0u32;

    for guard in guards {
        if guard.pop < depth {
            return false;
        }

        if cursor.pop((guard.pop - depth) as usize) != 0 {
            return false;
        }

        let Some(&state) = cursor.top() else {
            return false;
        };
        if guard.states.binary_search(&state).is_err() {
            return false;
        }

        depth = guard.pop;
    }

    true
}

#[inline]
fn virtual_stack_may_apply_guarded_shift(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    shift: &GuardedStackShift,
) -> bool {
    if !virtual_stack_satisfies_guards(stack, &shift.guards) {
        return false;
    }

    let mut cursor = stack.clone();
    cursor.pop(shift.pop as usize) == 0
}

fn reduce_sources_from_isolated(gss: &ParserGSS, rhs_len: usize) -> ReduceSources {
    let popped = gss.popn(rhs_len as isize);
    if popped.is_empty() {
        return SmallVec::new();
    }
    if let Some(v) = popped.single_top_value() {
        let mut result = SmallVec::new();
        result.push((v, popped));
        return result;
    }
    let top_vals = popped.peek_values();
    let mut result = SmallVec::new();
    for v in top_vals {
        result.push((v, popped.isolate(Some(v))));
    }
    result
}

fn reduce_branches_from_isolated(
    table: &GLRTable,
    gss: &ParserGSS,
    nt: u32,
    rhs_len: usize,
) -> ReduceBranches {
    if let Some(mut stack) = gss.try_virtual_stack() {
        if stack.pop(rhs_len) == 0 {
            if let Some(&goto_from) = stack.top() {
                if let Some((target, is_replace)) = table.goto_target(goto_from, nt) {
                    let mut branches = SmallVec::new();
                    branches.push((stack.into_gss(), target, is_replace));
                    return branches;
                }
            }
        }
    }

    let mut branches = SmallVec::new();
    for (goto_from, base) in reduce_sources_from_isolated(gss, rhs_len) {
        if let Some((target, is_replace)) = table.goto_target(goto_from, nt) {
            branches.push((base, target, is_replace));
        }
    }
    branches
}

fn merge_into(dst: &mut ParserGSS, branch: ParserGSS) {
    if branch.is_empty() {
        return;
    }
    if dst.is_empty() {
        *dst = branch;
    } else {
        *dst = dst.merge(&branch);
    }
}

fn advance_deterministically_from_vstack_raw(
    table: &GLRTable,
    mut stack: VirtualStack<u32, TerminalsDisallowed>,
    token: TerminalID,
) -> (AdvancedBranch, bool) {
    loop {
        let Some(&state) = stack.top() else {
            break;
        };

        match table.action(state, token) {
            Some(Action::Reduce(nt, len)) => {
                let rhs_len = *len as usize;
                if rhs_len < stack.len() {
                    if rhs_len == 1 {
                        if let Some(goto_from) = stack.parent_of_top() {
                            match table.goto_target(goto_from, *nt) {
                                Some((target, false)) if stack.replace_top(target) => continue,
                                Some((target, true)) => {
                                    stack.pop(2);
                                    stack.push(target);
                                    continue;
                                }
                                Some(_) | None => {
                                    return (AdvancedBranch::Gss(ParserGSS::empty()), false);
                                }
                            }
                        }
                    }

                    stack.pop(rhs_len);
                    let goto_from = *stack.top().unwrap();
                    match table.goto_target(goto_from, *nt) {
                        Some((target, false)) => stack.push(target),
                        Some((target, true)) => {
                            stack.replace_top(target);
                        }
                        None => {
                              return (AdvancedBranch::Gss(ParserGSS::empty()), false);
                        }
                    }
                } else {
                    let popped = stack.into_gss_after_popping(rhs_len);
                    let mut shifts = SmallVec::<[FloorCrossShift; 8]>::new();
                    for goto_from in popped.peek_values() {
                        if let Some((target, is_replace)) = table.goto_target(goto_from, *nt) {
                            shifts.push((goto_from, target, is_replace));
                        }
                    }
                    let rebuilt = rebuild_floor_cross_from_shifts(popped, shifts);
                      let Some(next_stack) = rebuilt.try_virtual_stack() else {
                          return (AdvancedBranch::Gss(rebuilt), false);
                      };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                } else {
                    stack.push(*target);
                }
                return (AdvancedBranch::Stack(stack), true);
            }
            Some(Action::StackShifts(shifts)) => {
                return (AdvancedBranch::Gss(apply_stack_shifts(stack.into_gss(), shifts)), true);
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                return (
                    AdvancedBranch::Gss(apply_guarded_stack_shifts_from_vstack(
                        stack,
                        shifts,
                        table.guarded_shift_index(state, token),
                    )),
                    true,
                );
            }
            Some(Action::Split { .. }) | Some(Action::Accept) | None => break,
        }
    }

    (AdvancedBranch::Stack(stack), false)
}

fn advance_deterministically_from_vstack(
    table: &GLRTable,
    stack: VirtualStack<u32, TerminalsDisallowed>,
    token: TerminalID,
) -> (ParserGSS, bool) {
    let (advanced, ok) = advance_deterministically_from_vstack_raw(table, stack, token);
    (advanced.into_gss(), ok)
}

fn advance_reduce_branch(
    table: &GLRTable,
    base: ParserGSS,
    target: u32,
    is_replace: bool,
    token: TerminalID,
) -> (AdvancedBranch, bool) {
    if let Some(mut stack) = base.try_virtual_stack() {
        if is_replace {
            stack.replace_top(target);
        } else {
            stack.push(target);
        }
        advance_deterministically_from_vstack_raw(table, stack, token)
    } else {
        let mut branch = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        let det_ok = advance_deterministically(table, &mut branch, token);
        (AdvancedBranch::Gss(branch), det_ok)
    }
}

fn single_concrete_path_as_vstack(
    gss: &ParserGSS,
) -> Option<VirtualStack<u32, TerminalsDisallowed>> {
    let mut top_first = SmallVec::<[u32; 16]>::new();
    let acc = gss.single_path_top_first_and_acc(&mut top_first)?;
    if top_first.len() > SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH {
        return None;
    }

    let mut stack = top_first.into_vec();
    stack.reverse();
    ParserGSS::from_single_stack(stack, acc).try_virtual_stack()
}

fn advance_split_from_vstack(
    table: &GLRTable,
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    token: TerminalID,
    shift: Option<(u32, bool)>,
    reduces: &[(u32, u32)],
) -> Option<ParserGSS> {
    let mut shifted = ParserGSS::empty();

    if let Some((target, is_replace)) = shift {
        let mut branch = stack.clone();
        if is_replace {
            branch.replace_top(target);
        } else {
            branch.push(target);
        }
        shifted = branch.into_gss();
    }

    for &(nt, len) in reduces {
        let mut branch = stack.clone();
        if branch.pop(len as usize) != 0 {
            return None;
        }
        let goto_from = *branch.top()?;
        let (target, is_replace) = table.goto_target(goto_from, nt)?;
        if is_replace {
            branch.replace_top(target);
        } else {
            branch.push(target);
        }

        let (branch, det_ok) = advance_deterministically_from_vstack_raw(table, branch, token);
        if !det_ok {
            return None;
        }
        match branch {
            AdvancedBranch::Stack(stack) => {
                shifted = shifted.absorb_vstack_same_acc_owned(stack);
            }
            AdvancedBranch::Gss(gss) => {
                merge_into(&mut shifted, gss);
            }
        }
    }

    Some(shifted)
}

fn advance_deterministically_profiled(
    table: &GLRTable,
    gss: &mut ParserGSS,
    token: TerminalID,
    profile: &mut AdvanceProfile,
) -> bool {
    let Some(mut stack) = gss
        .try_virtual_stack()
        .or_else(|| single_concrete_path_as_vstack(gss))
    else {
        profile.det_exit_reason = 6;
        return false;
    };

    profile.deterministic_entered = true;
    profile.vstack_len = stack.len() as u32;

    loop {
        let Some(&state) = stack.top() else {
            profile.det_exit_reason = 5;
            break;
        };

        profile.n_det_action_lookups += 1;
        let action = table.action(state, token);
        if let Some(trace) = profile.trace.as_mut() {
            let trace_gss = stack.clone().into_gss();
            trace.det_steps
                .push(trace_action_summary(table, state, &trace_gss, action));
        }
        match action {
            Some(Action::Reduce(nt, len)) => {
                let rhs_len = *len as usize;
                if rhs_len < stack.len() {
                    profile.n_reduces_above_floor += 1;
                    if rhs_len == 1 {
                        if let Some(goto_from) = stack.parent_of_top() {
                            profile.n_det_goto_lookups += 1;
                            match table.goto_target(goto_from, *nt) {
                                Some((target, false)) if stack.replace_top(target) => continue,
                                Some((target, true)) => {
                                    stack.pop(2);
                                    stack.push(target);
                                    continue;
                                }
                                Some(_) | None => {
                                    *gss = ParserGSS::empty();
                                    profile.det_exit_reason = 4;
                                    return false;
                                }
                            }
                        }
                    }

                    stack.pop(rhs_len);
                    let goto_from = *stack.top().unwrap();
                    profile.n_det_goto_lookups += 1;
                    match table.goto_target(goto_from, *nt) {
                        Some((target, false)) => stack.push(target),
                        Some((target, true)) => {
                            stack.replace_top(target);
                        }
                        None => {
                            *gss = ParserGSS::empty();
                            profile.det_exit_reason = 4;
                            return false;
                        }
                    }
                } else {
                    profile.n_floor_crossings += 1;
                    profile.n_det_popn_ops += 1;
                    let floor_cross_start = std::time::Instant::now();
                    let popped = stack.into_gss_after_popping(rhs_len);
                    let mut shifts = SmallVec::<[FloorCrossShift; 8]>::new();
                    for goto_from in popped.peek_values() {
                        profile.n_det_goto_lookups += 1;
                        if let Some((target, is_replace)) = table.goto_target(goto_from, *nt) {
                            shifts.push((goto_from, target, is_replace));
                        }
                    }
                    let rebuilt = rebuild_floor_cross_from_shifts(popped, shifts);
                    profile.det_floor_cross_ns += floor_cross_start.elapsed().as_nanos() as u64;
                    let Some(next_stack) = rebuilt.try_virtual_stack() else {
                        *gss = rebuilt;
                        profile.det_exit_reason = 7;
                        return false;
                    };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                } else {
                    stack.push(*target);
                }
                *gss = stack.into_gss();
                profile.det_exit_reason = 1;
                return true;
            }
            Some(Action::StackShifts(shifts)) => {
                *gss = apply_stack_shifts(stack.into_gss(), shifts);
                profile.det_exit_reason = 1;
                return true;
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                *gss = apply_guarded_stack_shifts_from_vstack(
                    stack,
                    shifts,
                    table.guarded_shift_index(state, token),
                );
                profile.det_exit_reason = 1;
                return true;
            }
            Some(Action::Split { shift, reduces, accept: false }) => {
                if let Some(shifted) =
                    advance_split_from_vstack(table, &stack, token, *shift, reduces)
                {
                    *gss = shifted;
                    profile.det_exit_reason = 1;
                    return true;
                }
                profile.det_exit_reason = 2;
                profile.det_exit_state = state;
                break;
            }
            Some(Action::Split { .. }) => {
                profile.det_exit_reason = 2;
                profile.det_exit_state = state;
                break;
            }
            Some(Action::Accept) => {
                profile.det_exit_reason = 3;
                profile.det_exit_state = state;
                break;
            }
            None => {
                profile.det_exit_reason = 4;
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
    use std::time::Instant;

    let mut shifted = ParserGSS::empty();
    loop {
        profile.n_nondet_waves += 1;
        let frontier_states = closure.peek_values();
        if let Some(trace) = profile.trace.as_mut() {
            trace.nondet_waves.push(AdvanceTraceWave {
                wave_index: profile.n_nondet_waves,
                frontier_states: frontier_states.to_vec(),
                branches: Vec::new(),
            });
        }

        if let Some(shifted_wave) =
            try_advance_single_active_pop1_reduce_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_reduce_ops += 1;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_pop1_common_base_reduce_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_reduce_ops += frontier_states.len() as u32;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_mixed_top_pop1_shift_reduce_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_reduce_ops += frontier_states
                .iter()
                .filter(|&&state| matches!(table.action(state, token), Some(Action::Reduce(..))))
                .count() as u32;
            profile.n_nondet_merges += frontier_states.len() as u32;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_pop1_stackshift_shift_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_pop1_reduce_plus_stackshift_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_reduce_ops += 1;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_single_alt_pop1_common_suffix_stackshift_wave(table, &closure, token)
        {
            if let Some(trace) = profile.trace.as_mut()
                && let Some(wave) = trace.nondet_waves.last_mut()
            {
                for &state in &frontier_states {
                    let branch_gss = closure.isolate(Some(state));
                    wave.branches.push(trace_action_summary(
                        table,
                        state,
                        &branch_gss,
                        table.action(state, token),
                    ));
                }
            }
            profile.n_nondet_branches += frontier_states.len() as u32;
            profile.n_nondet_merges += 1;
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        let mut next = ParserGSS::empty();

        for state in frontier_states {
            profile.n_nondet_branches += 1;
            let action = table.action(state, token);
            if let Some(trace) = profile.trace.as_mut() {
                let branch_gss = closure.isolate(Some(state));
                if let Some(wave) = trace.nondet_waves.last_mut() {
                    wave.branches
                        .push(trace_action_summary(table, state, &branch_gss, action));
                }
            }
            let Some(action) = action else {
                continue;
            };
            profile.n_nondet_isolates += 1;
            let pure_reduce = matches!(action, Action::Reduce(..));
            let mut isolated = closure.isolate(Some(state));
            match action {
                Action::Shift(target, is_replace) => {
                    profile.n_nondet_merges += 1;
                    if *is_replace {
                        shifted = shifted.absorb_push_same_acc(*target, &isolated.popn(1));
                    } else {
                        shifted = shifted.absorb_push_same_acc(*target, &isolated);
                    }
                    continue;
                }
                Action::StackShifts(shifts) => {
                    profile.n_nondet_merges += 1;
                    merge_into(&mut shifted, apply_stack_shifts(isolated, shifts));
                    continue;
                }
                Action::GuardedStackShifts(shifts) => {
                    profile.n_nondet_merges += 1;
                    let branch = if let Some(stack) = isolated.try_virtual_stack() {
                        apply_guarded_stack_shifts_from_vstack(
                            stack,
                            shifts,
                            table.guarded_shift_index(state, token),
                        )
                    } else {
                        apply_guarded_stack_shifts(
                            isolated,
                            shifts,
                            table.guarded_shift_index(state, token),
                        )
                    };
                    merge_into(&mut shifted, branch);
                    continue;
                }
                _ => {}
            }
            let reduce_base = if pure_reduce { None } else { Some(isolated.clone()) };
            if !pure_reduce {
                let det_start = Instant::now();
                if advance_deterministically(table, &mut isolated, token) {
                    profile.nondet_det_ns += det_start.elapsed().as_nanos() as u64;
                    profile.n_nondet_merges += 1;
                    merge_into(&mut shifted, isolated);
                    continue;
                }
                profile.nondet_det_ns += det_start.elapsed().as_nanos() as u64;
            }

            if let Some(target) = action.shift_target() {
                let is_replace = action.shift_is_replace() && !table.forwarded_shifts.contains(&(state, token));
                profile.n_nondet_merges += 1;
                if is_replace {
                    shifted = shifted.absorb_push_same_acc(target, &isolated.popn(1));
                } else {
                    shifted = shifted.absorb_push_same_acc(target, &isolated);
                }
            }

            if let Action::StackShifts(shifts) = action {
                profile.n_nondet_merges += 1;
                merge_into(&mut shifted, apply_stack_shifts(isolated.clone(), shifts));
            }

            action.for_each_reduce(|nt, len| {
                let reduce_base = reduce_base.as_ref().unwrap_or(&isolated);
                for (base, target, is_replace) in
                    reduce_branches_from_isolated(table, reduce_base, nt, len as usize)
                {
                    profile.n_nondet_reduce_ops += 1;
                    let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
                    if det_ok {
                        profile.n_nondet_merges += 1;
                        match branch {
                            AdvancedBranch::Stack(stack) => {
                                let current = std::mem::replace(&mut shifted, ParserGSS::empty());
                                shifted = current.absorb_vstack_same_acc_owned(stack);
                            }
                            AdvancedBranch::Gss(branch) => {
                                merge_into(&mut shifted, branch);
                            }
                        }
                    } else {
                        profile.n_nondet_merges += 1;
                        merge_into(&mut next, branch.into_gss());
                    }
                }
            });
        }

        if next.is_empty() {
            return shifted;
        }
        closure = next;
    }
}

fn advance_nondeterministically(
    table: &GLRTable,
    mut closure: ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    let mut shifted = ParserGSS::empty();

    loop {
        if let Some(shifted_wave) =
            try_advance_single_active_pop1_reduce_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_pop1_common_base_reduce_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_mixed_top_pop1_shift_reduce_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_pop1_stackshift_shift_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_pop1_reduce_plus_stackshift_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        if let Some(shifted_wave) =
            try_advance_single_alt_pop1_common_suffix_stackshift_wave(table, &closure, token)
        {
            merge_into(&mut shifted, shifted_wave);
            return shifted;
        }

        let mut next = ParserGSS::empty();

        for state in closure.peek_values() {
            let Some(action) = table.action(state, token) else {
                continue;
            };
            let pure_reduce = matches!(action, Action::Reduce(..));
            let mut isolated = closure.isolate(Some(state));
            match action {
                Action::Shift(target, is_replace) => {
                    if *is_replace {
                        shifted = shifted.absorb_push_same_acc(*target, &isolated.popn(1));
                    } else {
                        shifted = shifted.absorb_push_same_acc(*target, &isolated);
                    }
                    continue;
                }
                Action::StackShifts(shifts) => {
                    merge_into(&mut shifted, apply_stack_shifts(isolated, shifts));
                    continue;
                }
                Action::GuardedStackShifts(shifts) => {
                    let branch = if let Some(stack) = isolated.try_virtual_stack() {
                        apply_guarded_stack_shifts_from_vstack(
                            stack,
                            shifts,
                            table.guarded_shift_index(state, token),
                        )
                    } else {
                        apply_guarded_stack_shifts(
                            isolated,
                            shifts,
                            table.guarded_shift_index(state, token),
                        )
                    };
                    merge_into(&mut shifted, branch);
                    continue;
                }
                _ => {}
            }
            let reduce_base = if pure_reduce { None } else { Some(isolated.clone()) };
            if !pure_reduce && advance_deterministically(table, &mut isolated, token) {
                merge_into(&mut shifted, isolated);
                continue;
            }

            if let Some(target) = action.shift_target() {
                let is_replace = action.shift_is_replace() && !table.forwarded_shifts.contains(&(state, token));
                if is_replace {
                    shifted = shifted.absorb_push_same_acc(target, &isolated.popn(1));
                } else {
                    shifted = shifted.absorb_push_same_acc(target, &isolated);
                }
            }

            if let Action::StackShifts(shifts) = action {
                merge_into(&mut shifted, apply_stack_shifts(isolated.clone(), shifts));
            }

            action.for_each_reduce(|nt, len| {
                let reduce_base = reduce_base.as_ref().unwrap_or(&isolated);
                for (base, target, is_replace) in
                    reduce_branches_from_isolated(table, reduce_base, nt, len as usize)
                {
                    let (branch, det_ok) = advance_reduce_branch(table, base, target, is_replace, token);
                    if det_ok {
                        match branch {
                            AdvancedBranch::Stack(stack) => {
                                let current = std::mem::replace(&mut shifted, ParserGSS::empty());
                                shifted = current.absorb_vstack_same_acc_owned(stack);
                            }
                            AdvancedBranch::Gss(branch) => {
                                merge_into(&mut shifted, branch);
                            }
                        }
                    } else {
                        merge_into(&mut next, branch.into_gss());
                    }
                }
            });
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
/// degenerates to an ordinary flat parse stack. This applies the textbook
/// LR reduce loop directly: inspect the top state's action, pop |rhs|
/// symbols, push the goto target, repeat - until a non-reduce action is
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
    let Some(mut stack) = gss
        .try_virtual_stack()
        .or_else(|| single_concrete_path_as_vstack(gss))
    else {
        return false;
    };

    loop {
        let Some(&state) = stack.top() else {
            break;
        };

        match table.action(state, token) {
            Some(Action::Reduce(nt, len)) => {
                let rhs_len = *len as usize;
                if rhs_len < stack.len() {
                    if rhs_len == 1 {
                        if let Some(goto_from) = stack.parent_of_top() {
                            match table.goto_target(goto_from, *nt) {
                                Some((target, false)) if stack.replace_top(target) => continue,
                                Some((target, true)) => {
                                    stack.pop(2);
                                    stack.push(target);
                                    continue;
                                }
                                Some(_) | None => {
                                    *gss = ParserGSS::empty();
                                    return false;
                                }
                            }
                        }
                    }

                    stack.pop(rhs_len);
                    let goto_from = *stack.top().unwrap();
                    match table.goto_target(goto_from, *nt) {
                        Some((target, false)) => stack.push(target),
                        Some((target, true)) => {
                            stack.replace_top(target);
                        }
                        None => {
                            *gss = ParserGSS::empty();
                            return false;
                        }
                    }
                } else {
                    let popped = stack.into_gss_after_popping(rhs_len);
                    let mut shifts = SmallVec::<[FloorCrossShift; 8]>::new();
                    for goto_from in popped.peek_values() {
                        if let Some((target, is_replace)) = table.goto_target(goto_from, *nt) {
                            shifts.push((goto_from, target, is_replace));
                        }
                    }
                    let rebuilt = rebuild_floor_cross_from_shifts(popped, shifts);
                    let Some(next_stack) = rebuilt.try_virtual_stack() else {
                        *gss = rebuilt;
                        return false;
                    };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                    *gss = stack.into_gss();
                } else {
                    stack.push(*target);
                    *gss = stack.into_gss();
                }
                return true;
            }
            Some(Action::StackShifts(shifts)) => {
                *gss = apply_stack_shifts(stack.into_gss(), shifts);
                return true;
            }
            Some(Action::GuardedStackShifts(shifts)) => {
                *gss = apply_guarded_stack_shifts_from_vstack(
                    stack,
                    shifts,
                    table.guarded_shift_index(state, token),
                );
                return true;
            }
            Some(Action::Split { shift, reduces, accept: false }) => {
                if let Some(shifted) =
                    advance_split_from_vstack(table, &stack, token, *shift, reduces)
                {
                    *gss = shifted;
                    return true;
                }
                break;
            }
            Some(Action::Split { .. }) => break,
            Some(Action::Accept) => break,
            None => break,
        }
    }

    *gss = stack.into_gss();
    false
}

/// Precise predicate for whether this parser stack can advance on `token`.
///
/// Returns `true` if and only if at least one current parser path can definitely
/// advance on the given terminal. Returns `false` if no current parser path can
/// advance.
///
/// For `RowPresenceExact`, `advance` was captured from the precise parser row
/// before stack-effect lowering. Guarded stack shifts therefore select the exact
/// execution effect; they do not weaken admission. `ExactSimulation` tables use
/// reduction/guard simulation instead.
///
/// TODO: Rename this eventually, e.g. to `stack_can_advance_on`. The current
/// `may_advance` name sounds like a speculative approximation, but this is an
/// exact applicability predicate.
pub(crate) fn stack_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    if table.admission_policy == AdmissionPolicy::ExactSimulation {
        return exact_admission_may_advance_on(table, stack, token);
    }

    let admitted = stack
        .peek_values()
        .into_iter()
        .any(|state| table.advance_row_allows(state, token));
    if assert_row_presence_exact_enabled() {
        let simulated = exact_admission_may_advance_on(table, stack, token);
        assert_eq!(
            admitted,
            simulated,
            "RowPresenceExact mismatch for terminal {token}: construction={:?} stacks={:?}",
            table.construction,
            stack.to_stacks(),
        );
    }
    admitted
}

fn exact_admission_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    let mut queue = VecDeque::<ParserGSS>::new();
    let mut visited = FxHashSet::<Vec<Vec<u32>>>::default();

    for state in stack.peek_values() {
        exact_admission_enqueue_frontier(stack.isolate(Some(state)), &mut queue, &mut visited);
    }

    while let Some(frontier) = queue.pop_front() {
        for source_state in frontier.peek_values() {
            if !table.advance_row_allows(source_state, token) {
                continue;
            }
            let isolated = frontier.isolate(Some(source_state));
            let Some(action) = table.action(source_state, token) else {
                continue;
            };
            match action {
                Action::Shift(..) => return true,
                Action::StackShifts(shifts) => {
                    if !apply_stack_shifts(isolated.clone(), shifts).is_empty() {
                        return true;
                    }
                }
                Action::GuardedStackShifts(shifts) => {
                    if stack_may_apply_guarded_shifts(&isolated, shifts) {
                        return true;
                    }
                }
                Action::Reduce(nt, len) => {
                    exact_admission_enqueue_reduce(
                        table,
                        &isolated,
                        *nt,
                        *len as usize,
                        &mut queue,
                        &mut visited,
                    );
                }
                Action::Split {
                    shift,
                    reduces,
                    accept,
                } => {
                    if *accept && token == EOF {
                        return true;
                    }
                    if shift.is_some() {
                        return true;
                    }
                    for &(nt, len) in reduces {
                        exact_admission_enqueue_reduce(
                            table,
                            &isolated,
                            nt,
                            len as usize,
                            &mut queue,
                            &mut visited,
                        );
                    }
                }
                Action::Accept => {
                    if token == EOF {
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn exact_admission_may_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    if terminals.is_empty() {
        return false;
    }

    let mut queue = VecDeque::<(ParserGSS, BitSet)>::new();
    let mut visited = FxHashMap::<Vec<Vec<u32>>, BitSet>::default();

    for state in stack.peek_values() {
        exact_admission_enqueue_frontier_any(
            stack.isolate(Some(state)),
            terminals,
            &mut queue,
            &mut visited,
        );
    }

    while let Some((frontier, frontier_terminals)) = queue.pop_front() {
        for source_state in frontier.peek_values() {
            if !table.advance_row_intersects(source_state, &frontier_terminals) {
                continue;
            }

            let isolated = frontier.isolate(Some(source_state));
            let mut pending_reduces = SmallVec::<[(u32, u32, BitSet); 8]>::new();

            if exact_admission_for_each_matching_action(
                table,
                source_state,
                &frontier_terminals,
                |terminal, terminal_bit, action| {
                    exact_admission_process_action_any(
                        &isolated,
                        terminal,
                        terminal_bit,
                        action,
                        frontier_terminals.len(),
                        &mut pending_reduces,
                    )
                },
            ) {
                return true;
            }

            for (nt, len, reduce_terminals) in pending_reduces {
                exact_admission_enqueue_reduce_any(
                    table,
                    &isolated,
                    nt,
                    len as usize,
                    &reduce_terminals,
                    &mut queue,
                    &mut visited,
                );
            }
        }
    }

    false
}

fn exact_admission_for_each_matching_action(
    table: &GLRTable,
    state: u32,
    terminals: &BitSet,
    mut f: impl FnMut(TerminalID, usize, &Action) -> bool,
) -> bool {
    let Some(row) = table.action.get(state as usize) else {
        return false;
    };

    let active_count = terminals.count_ones();
    if active_count == 0 {
        return false;
    }

    if row.len() <= active_count {
        for (terminal, action) in row {
            let Some(bit) = exact_admission_terminal_bit(table, terminal, terminals.len()) else {
                continue;
            };
            if !terminals.contains(bit) || !table.advance_row_allows(state, terminal) {
                continue;
            }
            if f(terminal, bit, action) {
                return true;
            }
        }
        return false;
    }

    for bit in terminals.iter_ones() {
        let Some(terminal) = exact_admission_terminal_from_bit(table, bit) else {
            continue;
        };
        if !table.advance_row_allows(state, terminal) {
            continue;
        }
        if let Some(action) = table.action(state, terminal)
            && f(terminal, bit, action)
        {
            return true;
        }
    }

    false
}

fn exact_admission_process_action_any(
    isolated: &ParserGSS,
    terminal: TerminalID,
    terminal_bit: usize,
    action: &Action,
    terminals_len: usize,
    pending_reduces: &mut SmallVec<[(u32, u32, BitSet); 8]>,
) -> bool {
    match action {
        Action::Shift(..) => true,
        Action::StackShifts(shifts) => {
            !apply_stack_shifts(isolated.clone(), shifts).is_empty()
        }
        Action::GuardedStackShifts(shifts) => stack_may_apply_guarded_shifts(isolated, shifts),
        Action::Reduce(nt, len) => {
            exact_admission_add_pending_reduce(
                pending_reduces,
                *nt,
                *len,
                terminal_bit,
                terminals_len,
            );
            false
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            if *accept && terminal == EOF {
                return true;
            }
            if shift.is_some() {
                return true;
            }
            for &(nt, len) in reduces {
                exact_admission_add_pending_reduce(
                    pending_reduces,
                    nt,
                    len,
                    terminal_bit,
                    terminals_len,
                );
            }
            false
        }
        Action::Accept => terminal == EOF,
    }
}

fn exact_admission_add_pending_reduce(
    pending_reduces: &mut SmallVec<[(u32, u32, BitSet); 8]>,
    nt: u32,
    rhs_len: u32,
    terminal_bit: usize,
    terminals_len: usize,
) {
    if let Some((_, _, terminals)) = pending_reduces
        .iter_mut()
        .find(|(pending_nt, pending_len, _)| *pending_nt == nt && *pending_len == rhs_len)
    {
        terminals.set(terminal_bit);
        return;
    }

    let mut terminals = BitSet::new(terminals_len);
    terminals.set(terminal_bit);
    pending_reduces.push((nt, rhs_len, terminals));
}

fn exact_admission_enqueue_reduce_any(
    table: &GLRTable,
    isolated: &ParserGSS,
    nt: u32,
    rhs_len: usize,
    terminals: &BitSet,
    queue: &mut VecDeque<(ParserGSS, BitSet)>,
    visited: &mut FxHashMap<Vec<Vec<u32>>, BitSet>,
) {
    for (base, target, is_replace) in
        reduce_branches_from_isolated(table, isolated, nt, rhs_len)
    {
        let next = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        exact_admission_enqueue_frontier_any(next, terminals, queue, visited);
    }
}

fn exact_admission_enqueue_frontier_any(
    frontier: ParserGSS,
    terminals: &BitSet,
    queue: &mut VecDeque<(ParserGSS, BitSet)>,
    visited: &mut FxHashMap<Vec<Vec<u32>>, BitSet>,
) {
    if frontier.is_empty() || terminals.is_empty() {
        return;
    }

    let mut key: Vec<Vec<u32>> = frontier
        .to_stacks()
        .into_iter()
        .map(|(stack, _)| stack)
        .collect();
    key.sort();
    key.dedup();

    let new_terminals = if let Some(seen) = visited.get_mut(&key) {
        let delta = terminals.difference(seen);
        if delta.is_empty() {
            return;
        }
        seen.union_with(&delta);
        delta
    } else {
        visited.insert(key, terminals.clone());
        terminals.clone()
    };

    queue.push_back((frontier, new_terminals));
}

#[inline]
fn exact_admission_terminal_bit(
    table: &GLRTable,
    terminal: TerminalID,
    terminals_len: usize,
) -> Option<usize> {
    let bit = if terminal == EOF {
        table.num_terminals as usize
    } else if terminal < table.num_terminals {
        terminal as usize
    } else {
        return None;
    };
    (bit < terminals_len).then_some(bit)
}

#[inline]
fn exact_admission_terminal_from_bit(table: &GLRTable, bit: usize) -> Option<TerminalID> {
    if bit == table.num_terminals as usize {
        Some(EOF)
    } else if bit < table.num_terminals as usize {
        Some(bit as TerminalID)
    } else {
        None
    }
}

fn exact_admission_enqueue_reduce(
    table: &GLRTable,
    isolated: &ParserGSS,
    nt: u32,
    rhs_len: usize,
    queue: &mut VecDeque<ParserGSS>,
    visited: &mut FxHashSet<Vec<Vec<u32>>>,
) {
    for (base, target, is_replace) in
        reduce_branches_from_isolated(table, isolated, nt, rhs_len)
    {
        let next = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        exact_admission_enqueue_frontier(next, queue, visited);
    }
}

fn exact_admission_enqueue_frontier(
    frontier: ParserGSS,
    queue: &mut VecDeque<ParserGSS>,
    visited: &mut FxHashSet<Vec<Vec<u32>>>,
) {
    if frontier.is_empty() {
        return;
    }
    let mut key: Vec<Vec<u32>> = frontier
        .to_stacks()
        .into_iter()
        .map(|(stack, _)| stack)
        .collect();
    key.sort();
    key.dedup();
    if visited.insert(key) {
        queue.push_back(frontier);
    }
}

fn stack_may_apply_guarded_shifts(stack: &ParserGSS, shifts: &[GuardedStackShift]) -> bool {
    if let Some(virtual_stack) = stack.try_virtual_stack()
        && !virtual_stack.has_hidden_floor_values()
    {
        return shifts
            .iter()
            .any(|shift| virtual_stack_may_apply_guarded_shift(&virtual_stack, shift));
    }

    stack.to_stacks().into_iter().any(|(stack_values, acc)| {
        let single = ParserGSS::from_single_stack(stack_values, acc);
        let Some(virtual_stack) = single.try_virtual_stack() else {
            return false;
        };
        shifts
            .iter()
            .any(|shift| virtual_stack_may_apply_guarded_shift(&virtual_stack, shift))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ParserGSS,
        advance_concrete_stacks_reference,
        advance_stacks,
        apply_guarded_stack_shifts_to_vstack,
        stack_may_advance_on,
        stack_may_advance_on_any,
        try_advance_pop1_reduce_plus_stackshift_wave,
    };
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::analysis::EOF;
    use crate::compiler::glr::table::testing::build_test_table;
    use crate::compiler::glr::table::{
        Action, AdmissionPolicy, GLRTable, GuardedStackShift, StackShift, StackShiftGuard,
    };
    use crate::ds::bitset::BitSet;
    use crate::ds::leveled_gss::Merge;

    #[test]
    fn concrete_advance_reference_applies_whole_stack_effect_atomically() {
        let token = 0;
        let table = build_test_table(
            2,
            1,
            &[
                &[],
                &[(
                    token,
                    Action::StackShifts(vec![StackShift {
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[]],
        );
        let before = ParserGSS::from_single_stack(
            vec![0, 1],
            TerminalsDisallowed::new(),
        );
        let expected = ParserGSS::from_single_stack(
            vec![7],
            TerminalsDisallowed::new(),
        );

        assert_eq!(
            advance_concrete_stacks_reference(&table, &before, token),
            expected,
        );
    }

    #[test]
    fn concrete_advance_reference_merges_accumulators_at_reduce_closure_join() {
        let token = 0;
        let nt = 0;
        let table = build_test_table(
            6,
            1,
            &[
                &[],
                &[],
                &[(token, Action::Reduce(nt, 1))],
                &[(token, Action::Reduce(nt, 1))],
                &[(token, Action::Shift(5, false))],
                &[],
            ],
            &[&[(nt, (4, false))], &[], &[], &[], &[], &[]],
        );
        let left_acc = TerminalsDisallowed::new().with_insert(10, 20);
        let right_acc = TerminalsDisallowed::new().with_insert(11, 21);
        let before = ParserGSS::from_stacks(&[
            (vec![0, 2], left_acc.clone()),
            (vec![0, 3], right_acc.clone()),
        ]);
        let expected = ParserGSS::from_single_stack(
            vec![0, 4, 5],
            left_acc.merge(&right_acc),
        );

        assert_eq!(
            advance_concrete_stacks_reference(&table, &before, token),
            expected,
        );
    }

    #[test]
    fn advance_stacks_matches_reduce_fanout_collapse_fast_path() {
        let token = 0;
        let nt = 0;
        let table = build_test_table(
            5,
            1,
            &[
                &[],
                &[],
                &[(token, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] }]))],
                &[(token, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] }]))],
                &[(token, Action::Reduce(nt, 1))],
            ],
            &[
                &[(nt, (2, false))],
                &[(nt, (3, false))],
                &[],
                &[],
                &[],
            ],
        );

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 4], acc.clone()),
            (vec![1, 4], acc),
        ]);
        let expected = ParserGSS::from_single_stack(vec![7], TerminalsDisallowed::new());

        assert_eq!(advance_stacks(&table, &before, token), expected);
    }

    #[test]
    fn advance_stacks_selective_pure_frontier_shift_keeps_only_actionable_top() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 134];
        action_rows[131] = vec![(
            token,
            Action::StackShifts(vec![StackShift {
                pop: 0,
                pushes: vec![96],
            }]),
        )];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();
        let goto_rows = vec![Vec::new(); 134];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();
        let table = build_test_table(134, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 133], acc),
        ]);
        let expected = ParserGSS::from_single_stack(
            vec![0_u32, 1, 17, 47, 74, 131, 96],
            TerminalsDisallowed::new(),
        );

        assert_eq!(advance_stacks(&table, &before, token), expected);
    }

    #[test]
    fn pop1_reduce_plus_stackshift_wave_fast_path_matches_snowplow_shape() {
        let token = 0;
        let nt = 0;
        let mut action_rows = vec![Vec::new(); 989];
        action_rows[655] = vec![(
            token,
            Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![975] }]),
        )];
        action_rows[659] = vec![(
            token,
            Action::StackShifts(vec![
                StackShift { pop: 1, pushes: vec![654] },
                StackShift { pop: 1, pushes: vec![988] },
            ]),
        )];
        action_rows[987] = vec![(token, Action::Reduce(nt, 1))];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();

        let mut goto_rows = vec![Vec::new(); 989];
        goto_rows[87] = vec![(nt, (659, true))];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();

        let table = build_test_table(989, 1, &action_refs, &goto_refs);
        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 987], acc.clone()),
            (vec![0_u32, 87, 655], acc),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 975], TerminalsDisallowed::new()),
            (vec![0_u32, 654], TerminalsDisallowed::new()),
            (vec![0_u32, 988], TerminalsDisallowed::new()),
        ]);

        let mut fast_stacks = try_advance_pop1_reduce_plus_stackshift_wave(&table, &before, token)
            .expect("fast path should match this wave")
            .to_stacks();
        let mut expected_stacks = expected.to_stacks();
        fast_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(fast_stacks, expected_stacks);

        let mut actual_stacks = advance_stacks(&table, &before, token).to_stacks();
        actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(actual_stacks, expected_stacks);
    }

    #[test]
    fn pop1_reduce_plus_stackshift_wave_rejects_cross_product_base() {
        let token = 0;
        let nt = 0;
        let mut action_rows = vec![Vec::new(); 989];
        action_rows[655] = vec![(
            token,
            Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![975] }]),
        )];
        action_rows[659] = vec![(
            token,
            Action::StackShifts(vec![
                StackShift { pop: 1, pushes: vec![654] },
                StackShift { pop: 1, pushes: vec![988] },
            ]),
        )];
        action_rows[987] = vec![(token, Action::Reduce(nt, 1))];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();

        let mut goto_rows = vec![Vec::new(); 989];
        goto_rows[87] = vec![(nt, (659, true))];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();

        let table = build_test_table(989, 1, &action_refs, &goto_refs);
        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 987], acc.clone()),
            (vec![1_u32, 87, 655], acc),
        ]);

        assert_eq!(
            try_advance_pop1_reduce_plus_stackshift_wave(&table, &before, token),
            None
        );
    }

    #[test]
    fn may_advance_consults_admission_rows_not_execution_actions() {
        let token = 0;
        let mut table = build_test_table(
            2,
            1,
            &[&[], &[(token, Action::Shift(1, false))]],
            &[&[], &[]],
        );
        table.advance[1].clear(token as usize);

        let stack = ParserGSS::from_single_stack(vec![1], TerminalsDisallowed::new());
        assert!(table.action(1, token).is_some());
        assert!(!stack_may_advance_on(&table, &stack, token));

        let mut terminals = BitSet::new(1);
        terminals.set(token as usize);
        assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn may_advance_rechecks_guarded_stack_shifts_against_concrete_stack() {
        let token = 0;
        let mut table = build_test_table(
            3,
            1,
            &[
                &[],
                &[],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![0],
                        }],
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());

        assert!(table.advance_row_allows(2, token));
        assert!(advance_stacks(&table, &stack, token).is_empty());
        assert!(!stack_may_advance_on(&table, &stack, token));

        let mut terminals = BitSet::new(1);
        terminals.set(token as usize);
        assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn row_presence_admission_does_not_recheck_lowered_guarded_effects() {
        let token = 0;
        let table = build_test_table(
            3,
            1,
            &[
                &[],
                &[],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![0],
                        }],
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[], &[]],
        );
        let stack = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());

        assert_eq!(table.admission_policy, AdmissionPolicy::RowPresenceExact);
        assert!(table.advance_row_allows(2, token));
        assert!(stack_may_advance_on(&table, &stack, token));

        let mut terminals = BitSet::new(1);
        terminals.set(token as usize);
        assert!(stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn guarded_stack_shift_advance_distributes_over_merged_branched_floor() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 329];
        action_rows[74] = vec![(
            token,
            Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![171],
                    }],
                    pop: 2,
                    pushes: vec![213, 265],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![323],
                    }],
                    pop: 2,
                    pushes: vec![328, 370],
                },
            ]),
        )];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();
        let goto_rows = vec![Vec::new(); 329];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(329, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let left = ParserGSS::from_single_stack(vec![0, 171, 74], acc.clone());
        let right = ParserGSS::from_single_stack(vec![0, 323, 74], acc);
        let merged = left.merge(&right);

        let expected = advance_stacks(&table, &left, token)
            .merge(&advance_stacks(&table, &right, token));
        let actual = advance_stacks(&table, &merged, token);

        let mut expected_stacks = expected.to_stacks();
        let mut actual_stacks = actual.to_stacks();
        expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(actual_stacks, expected_stacks);
        assert!(stack_may_advance_on(&table, &merged, token));
    }

    #[test]
    fn stack_shift_advance_distributes_over_merged_branched_floor() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 75];
        action_rows[74] = vec![(
            token,
            Action::StackShifts(vec![StackShift {
                pop: 2,
                pushes: vec![265],
            }]),
        )];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(Vec::as_slice).collect();
        let goto_rows = vec![Vec::new(); 75];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(Vec::as_slice).collect();
        let table = build_test_table(75, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let left = ParserGSS::from_single_stack(vec![0, 171, 74], acc.clone());
        let right = ParserGSS::from_single_stack(vec![0, 323, 74], acc);
        let merged = left.merge(&right);

        let expected = advance_stacks(&table, &left, token)
            .merge(&advance_stacks(&table, &right, token));
        let actual = advance_stacks(&table, &merged, token);

        let mut expected_stacks = expected.to_stacks();
        let mut actual_stacks = actual.to_stacks();
        expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(actual_stacks, expected_stacks);
    }

    fn assert_advance_distributes_over_merge(
        table: &GLRTable,
        left: &ParserGSS,
        right: &ParserGSS,
        token: u32,
        case: &str,
    ) {
        let merged = left.merge(right);
        let expected = advance_stacks(table, left, token)
            .merge(&advance_stacks(table, right, token));
        let actual = advance_stacks(table, &merged, token);
        let concrete_reference = advance_concrete_stacks_reference(table, &merged, token);

        let mut expected_stacks = expected.to_stacks();
        let mut actual_stacks = actual.to_stacks();
        let mut reference_stacks = concrete_reference.to_stacks();
        expected_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        actual_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        reference_stacks.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(actual_stacks, expected_stacks, "{case}");
        assert_eq!(actual_stacks, reference_stacks, "{case}");
    }

    fn branched_floor_pair(common_suffix_len: usize) -> (ParserGSS, ParserGSS) {
        let suffix = [10_u32, 11, 12];
        let mut left = vec![0, 1];
        left.extend_from_slice(&suffix[..common_suffix_len]);
        let mut right = vec![0, 2];
        right.extend_from_slice(&suffix[..common_suffix_len]);
        let acc = TerminalsDisallowed::new();
        (
            ParserGSS::from_single_stack(left, acc.clone()),
            ParserGSS::from_single_stack(right, acc),
        )
    }

    fn assert_advance_matches_concrete_reference_case(
        table: &GLRTable,
        before: &ParserGSS,
        token: u32,
        case: &str,
    ) {
        let actual = advance_stacks(table, before, token);
        let expected = advance_concrete_stacks_reference(table, before, token);
        assert_eq!(
            super::normalized_concrete_stacks(&actual),
            super::normalized_concrete_stacks(&expected),
            "{case}: before={:?}",
            before.to_stacks(),
        );
    }

    #[test]
    fn generated_mixed_pop1_frontier_actions_match_concrete_reference() {
        let token = 0;
        let nt0 = 0;
        let nt1 = 1;
        let actions = vec![
            None,
            Some(Action::Shift(40, false)),
            Some(Action::Shift(41, true)),
            Some(Action::StackShifts(vec![StackShift {
                pop: 0,
                pushes: vec![42],
            }])),
            Some(Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![43],
            }])),
            Some(Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![44, 45],
            }])),
            Some(Action::StackShifts(vec![StackShift {
                pop: 2,
                pushes: vec![46],
            }])),
            Some(Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![47],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![48],
                },
            ])),
            Some(Action::Reduce(nt0, 1)),
            Some(Action::Reduce(nt1, 1)),
            Some(Action::Split {
                shift: Some((49, false)),
                reduces: vec![(nt0, 1)],
                accept: false,
            }),
            Some(Action::Split {
                shift: None,
                reduces: vec![(nt0, 1), (nt1, 1)],
                accept: false,
            }),
            Some(Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![10],
                }],
                pop: 2,
                pushes: vec![52],
            }])),
        ];

        for (left_index, left_action) in actions.iter().enumerate() {
            for (right_index, right_action) in actions.iter().enumerate() {
                let mut action_rows = vec![Vec::new(); 64];
                if let Some(action) = left_action {
                    action_rows[20].push((token, action.clone()));
                }
                if let Some(action) = right_action {
                    action_rows[21].push((token, action.clone()));
                }
                action_rows[30].push((token, Action::Shift(50, false)));
                action_rows[31].push((
                    token,
                    Action::StackShifts(vec![StackShift {
                        pop: 1,
                        pushes: vec![51],
                    }]),
                ));
                let action_refs: Vec<&[(u32, Action)]> =
                    action_rows.iter().map(Vec::as_slice).collect();

                let mut goto_rows = vec![Vec::new(); 64];
                goto_rows[10].push((nt0, (30, false)));
                goto_rows[10].push((nt1, (31, false)));
                let goto_refs: Vec<&[(u32, (u32, bool))]> =
                    goto_rows.iter().map(Vec::as_slice).collect();
                let table = build_test_table(64, 1, &action_refs, &goto_refs);

                let left_acc = TerminalsDisallowed::new().with_insert(60, 0);
                let right_acc = TerminalsDisallowed::new().with_insert(61, 0);
                let before = ParserGSS::from_stacks(&[
                    (vec![0, 10, 20], left_acc),
                    (vec![0, 10, 21], right_acc),
                ]);
                let case = format!(
                    "left_index={left_index} left={left_action:?} right_index={right_index} right={right_action:?}",
                );
                assert_advance_matches_concrete_reference_case(&table, &before, token, &case);
            }
        }
    }

    #[test]
    fn generated_top_action_advance_distributes_over_branched_floor_merge() {
        let token = 0;
        for common_suffix_len in 1..=3 {
            let (left, right) = branched_floor_pair(common_suffix_len);
            let top = 9 + common_suffix_len as u32;
            let mut actions = vec![
                Action::Shift(40, false),
                Action::Shift(40, true),
            ];

            for pop in 0..=common_suffix_len + 1 {
                actions.push(Action::StackShifts(vec![StackShift {
                    pop: pop as u32,
                    pushes: vec![40],
                }]));
                actions.push(Action::StackShifts(vec![StackShift {
                    pop: pop as u32,
                    pushes: Vec::new(),
                }]));
            }
            actions.push(Action::StackShifts(vec![
                StackShift {
                    pop: 0,
                    pushes: vec![40],
                },
                StackShift {
                    pop: common_suffix_len as u32 + 1,
                    pushes: vec![41],
                },
            ]));
            actions.push(Action::StackShifts(vec![
                StackShift {
                    pop: common_suffix_len as u32 + 1,
                    pushes: vec![40],
                },
                StackShift {
                    pop: 0,
                    pushes: vec![41],
                },
            ]));
            actions.push(Action::StackShifts(vec![
                StackShift {
                    pop: common_suffix_len as u32,
                    pushes: vec![40, 42],
                },
                StackShift {
                    pop: common_suffix_len as u32,
                    pushes: vec![41, 42],
                },
            ]));

            actions.push(Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: common_suffix_len as u32,
                        states: vec![1],
                    }],
                    pop: common_suffix_len as u32 + 1,
                    pushes: vec![40],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: common_suffix_len as u32,
                        states: vec![2],
                    }],
                    pop: common_suffix_len as u32 + 1,
                    pushes: vec![41],
                },
            ]));

            for (action_index, action) in actions.into_iter().enumerate() {
                let mut action_rows = vec![Vec::new(); 64];
                action_rows[top as usize] = vec![(token, action)];
                let action_refs: Vec<&[(u32, Action)]> =
                    action_rows.iter().map(Vec::as_slice).collect();
                let goto_rows = vec![Vec::new(); 64];
                let goto_refs: Vec<&[(u32, (u32, bool))]> =
                    goto_rows.iter().map(Vec::as_slice).collect();
                let table = build_test_table(64, 1, &action_refs, &goto_refs);
                let case = format!(
                    "common_suffix_len={common_suffix_len} action_index={action_index} action={:?}",
                    table.action(top, token),
                );
                assert_advance_distributes_over_merge(&table, &left, &right, token, &case);
            }
        }
    }

    #[test]
    fn generated_reduce_chain_advance_distributes_over_branched_floor_merge() {
        let token = 0;
        let nt = 0;
        for common_suffix_len in 1..=3 {
            let (left, right) = branched_floor_pair(common_suffix_len);
            let top = 9 + common_suffix_len as u32;
            for reduce_len in 1..=common_suffix_len {
                for is_replace in [false, true] {
                    let mut action_rows = vec![Vec::new(); 64];
                    action_rows[top as usize] =
                        vec![(token, Action::Reduce(nt, reduce_len as u32))];
                    action_rows[50] = vec![(token, Action::Shift(60, false))];
                    action_rows[51] = vec![(token, Action::Shift(60, false))];

                    let left_stacks = left.to_stacks();
                    let right_stacks = right.to_stacks();
                    let left_values = &left_stacks[0].0;
                    let right_values = &right_stacks[0].0;
                    let left_goto_from = left_values[left_values.len() - reduce_len - 1];
                    let right_goto_from = right_values[right_values.len() - reduce_len - 1];

                    let mut goto_rows = vec![Vec::new(); 64];
                    goto_rows[left_goto_from as usize].push((nt, (50, is_replace)));
                    if right_goto_from != left_goto_from {
                        goto_rows[right_goto_from as usize].push((nt, (51, is_replace)));
                    }

                    let action_refs: Vec<&[(u32, Action)]> =
                        action_rows.iter().map(Vec::as_slice).collect();
                    let goto_refs: Vec<&[(u32, (u32, bool))]> =
                        goto_rows.iter().map(Vec::as_slice).collect();
                    let table = build_test_table(64, 1, &action_refs, &goto_refs);
                    let case = format!(
                        "reduce common_suffix_len={common_suffix_len} reduce_len={reduce_len} is_replace={is_replace} left_goto={left_goto_from} right_goto={right_goto_from}",
                    );
                    assert_advance_distributes_over_merge(&table, &left, &right, token, &case);
                }
            }
        }
    }

    #[test]
    fn exact_admission_rejects_union_reduce_with_no_real_goto() {
        let token = 0;
        let nt = 0;
        let mut table = build_test_table(
            3,
            1,
            &[&[], &[], &[(token, Action::Reduce(nt, 1))]],
            &[&[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

        assert!(!stack_may_advance_on(&table, &stack, token));
    }

    #[test]
    fn exact_admission_accepts_reduce_goto_then_shift_path() {
        let token = 0;
        let nt = 0;
        let mut table = build_test_table(
            5,
            1,
            &[
                &[],
                &[],
                &[(token, Action::Reduce(nt, 1))],
                &[(token, Action::Shift(4, false))],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

        assert!(stack_may_advance_on(&table, &stack, token));
    }

    #[test]
    fn exact_admission_any_uses_same_exactness_as_single_terminal() {
        let token = 0;
        let nt = 0;
        let mut table = build_test_table(
            5,
            2,
            &[
                &[],
                &[],
                &[(token, Action::Reduce(nt, 1))],
                &[(token, Action::Shift(4, false))],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;
        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());

        let mut terminals = BitSet::new(3);
        terminals.set(token as usize);
        assert_eq!(
            stack_may_advance_on(&table, &stack, token),
            stack_may_advance_on_any(&table, &stack, &terminals)
        );
    }

    fn assert_exact_any_matches_disjunction(
        table: &GLRTable,
        stack: &ParserGSS,
        terminals: &BitSet,
    ) {
        let disjunction = terminals.iter_ones().any(|bit| match bit {
            bit if bit == table.num_terminals as usize => stack_may_advance_on(table, stack, EOF),
            bit if bit < table.num_terminals as usize => {
                stack_may_advance_on(table, stack, bit as u32)
            }
            _ => false,
        });

        assert_eq!(
            stack_may_advance_on_any(table, stack, terminals),
            disjunction
        );
    }

    #[test]
    fn exact_admission_any_does_not_mix_lookahead_reductions() {
        let reduce_token = 0;
        let shift_token = 1;
        let nt = 0;
        let mut table = build_test_table(
            5,
            2,
            &[
                &[],
                &[],
                &[(reduce_token, Action::Reduce(nt, 1))],
                &[(shift_token, Action::Shift(4, false))],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
        let mut terminals = BitSet::new(3);
        terminals.set(reduce_token as usize);
        terminals.set(shift_token as usize);

        assert!(!stack_may_advance_on(&table, &stack, reduce_token));
        assert!(!stack_may_advance_on(&table, &stack, shift_token));
        assert_exact_any_matches_disjunction(&table, &stack, &terminals);
        assert!(!stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn exact_admission_any_batches_reduce_frontier_by_terminal_set() {
        let token_a = 0;
        let token_b = 1;
        let nt = 0;
        let mut table = build_test_table(
            5,
            2,
            &[
                &[],
                &[],
                &[
                    (token_a, Action::Reduce(nt, 1)),
                    (token_b, Action::Reduce(nt, 1)),
                ],
                &[(token_b, Action::Shift(4, false))],
                &[],
            ],
            &[&[(nt, (3, false))], &[], &[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let stack = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
        let mut terminals = BitSet::new(3);
        terminals.set(token_a as usize);
        terminals.set(token_b as usize);

        assert!(!stack_may_advance_on(&table, &stack, token_a));
        assert!(stack_may_advance_on(&table, &stack, token_b));
        assert_exact_any_matches_disjunction(&table, &stack, &terminals);
        assert!(stack_may_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn exact_admission_any_preserves_guarded_shift_checks() {
        let token = 0;
        let other_token = 1;
        let mut table = build_test_table(
            3,
            2,
            &[
                &[],
                &[],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![0],
                        }],
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[], &[]],
        );
        table.admission_policy = AdmissionPolicy::ExactSimulation;

        let mut terminals = BitSet::new(3);
        terminals.set(token as usize);
        terminals.set(other_token as usize);

        let rejected = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());
        assert_exact_any_matches_disjunction(&table, &rejected, &terminals);
        assert!(!stack_may_advance_on_any(&table, &rejected, &terminals));

        let accepted = ParserGSS::from_single_stack(vec![0, 2], TerminalsDisallowed::new());
        assert_exact_any_matches_disjunction(&table, &accepted, &terminals);
        assert!(stack_may_advance_on_any(&table, &accepted, &terminals));
    }

    #[test]
    fn exact_admission_any_handles_eof_acceptance() {
        let token = 0;
        let mut table = build_test_table(1, 1, &[&[(EOF, Action::Accept)]], &[&[]]);
        table.admission_policy = AdmissionPolicy::ExactSimulation;
        let stack = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());

        let mut non_eof = BitSet::new(2);
        non_eof.set(token as usize);
        assert_exact_any_matches_disjunction(&table, &stack, &non_eof);
        assert!(!stack_may_advance_on_any(&table, &stack, &non_eof));

        let mut eof = BitSet::new(2);
        eof.set(table.num_terminals as usize);
        assert_exact_any_matches_disjunction(&table, &stack, &eof);
        assert!(stack_may_advance_on_any(&table, &stack, &eof));
    }

    #[test]
    fn advance_stacks_materializes_single_concrete_path_for_split() {
        let token = 0;
        let nt = 0;
        let table = build_test_table(
            6,
            1,
            &[
                &[],
                &[],
                &[(token, Action::Split {
                    shift: Some((3, false)),
                    reduces: vec![(nt, 1)],
                    accept: false,
                })],
                &[],
                &[(token, Action::Shift(5, false))],
                &[],
            ],
            &[
                &[(nt, (4, false))],
                &[],
                &[],
                &[],
                &[],
                &[],
            ],
        );

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1], acc.clone()),
            (vec![0, 2], acc.clone()),
        ])
        .popn(1)
        .push(2);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 2, 3], acc.clone()),
            (vec![0, 4, 5], acc),
        ]);

        let mut actual_stacks = advance_stacks(&table, &before, token).to_stacks();
        let mut expected_stacks = expected.to_stacks();
        actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));

        assert_eq!(actual_stacks, expected_stacks);
    }

    #[test]
    fn indexed_guarded_vstack_matches_linear_guarded_vstack() {
        let token = 0;
        let mut table = build_test_table(
            1,
            1,
            &[&[(
                token,
                Action::GuardedStackShifts(vec![
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![10, 20],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![1],
                            },
                        ],
                        pop: 3,
                        pushes: vec![50],
                    },
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![10],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![2],
                            },
                        ],
                        pop: 3,
                        pushes: vec![51],
                    },
                    GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![10, 20],
                        }],
                        pop: 2,
                        pushes: vec![52],
                    },
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![30],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![1],
                            },
                        ],
                        pop: 3,
                        pushes: vec![53],
                    },
                ]),
            )]],
            &[&[]],
        );
        table.rebuild_guarded_shift_index();

        let shifts = match table.action(0, token) {
            Some(Action::GuardedStackShifts(shifts)) => shifts,
            other => panic!("expected guarded stack shifts, got {other:?}"),
        };
        let index = table
            .guarded_shift_index(0, token)
            .expect("expected guarded shift index");

        let stack_a = ParserGSS::from_single_stack(vec![0, 1, 10, 99], TerminalsDisallowed::new());
        let stack_b = ParserGSS::from_single_stack(vec![0, 2, 10, 99], TerminalsDisallowed::new());

        for stack in [&stack_a, &stack_b] {
            let vstack = stack.try_virtual_stack().expect("expected virtual stack");
            let mut indexed = apply_guarded_stack_shifts_to_vstack(&vstack, shifts, Some(index)).to_stacks();
            let mut linear = apply_guarded_stack_shifts_to_vstack(&vstack, shifts, None).to_stacks();
            indexed.sort_by(|left, right| left.0.cmp(&right.0));
            linear.sort_by(|left, right| left.0.cmp(&right.0));
            assert_eq!(indexed, linear);
        }
    }
}

/// Precise predicate for whether this parser stack can advance on any terminal in
/// `terminals`.
///
/// Returns `true` if and only if at least one current parser path can definitely
/// advance on one of the given terminals. Returns `false` if no current parser
/// path can advance on any of them.
///
/// For `RowPresenceExact`, `advance` was captured from the precise parser row
/// before stack-effect lowering. Guarded stack shifts therefore select the exact
/// execution effect; they do not weaken admission. `ExactSimulation` tables use
/// reduction/guard simulation instead.
///
/// TODO: Rename this eventually, e.g. to `stack_can_advance_on_any`. The current
/// `may_advance` name sounds like a speculative approximation, but this is an
/// exact applicability predicate.
pub(crate) fn stack_may_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    if table.admission_policy == AdmissionPolicy::ExactSimulation {
        return exact_admission_may_advance_on_any(table, stack, terminals);
    }

    let admitted = stack
        .peek_values()
        .into_iter()
        .any(|state| table.advance_row_intersects(state, terminals));
    if assert_row_presence_exact_enabled() {
        let simulated = exact_admission_may_advance_on_any(table, stack, terminals);
        assert_eq!(
            admitted,
            simulated,
            "RowPresenceExact mismatch for terminal set {:?}: construction={:?} stacks={:?}",
            terminals.iter_ones().collect::<Vec<_>>(),
            table.construction,
            stack.to_stacks(),
        );
    }
    admitted
}

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    let has_eof_action = stack
        .peek_values()
        .iter()
        .any(|&state| table.action(state, EOF).is_some());

    has_eof_action
}
