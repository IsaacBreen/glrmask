use super::accumulator::TerminalsDisallowed;
use super::analysis::EOF;
use super::table::{Action, GLRTable, GuardedStackShift, StackShift, StackShiftGuard};
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::{LeveledGSS, VirtualStack};
use crate::grammar::flat::TerminalID;
use smallvec::SmallVec;
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

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
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
    advance_stacks_core(table, stack.clone(), token)
}

/// Like `advance_stacks` but takes ownership of the GSS, avoiding an
/// unnecessary Arc clone when the caller doesn't need the original.
pub(crate) fn advance_stacks_owned(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_core(table, stack, token)
}

pub(crate) fn advance_stacks_profiled(
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
                let shifted = apply_guarded_stack_shifts(gss, shifts);
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
            return apply_guarded_stack_shifts(gss, shifts);
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
    Some(gss.apply_top_pure_shifts(shifts))
}

fn try_advance_single_alt_pop1_common_suffix_stackshift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if !(2..=3).contains(&states.len()) {
        return None;
    }

    let mut per_state_prefixes: SmallVec<[(u32, u32, bool); 3]> = SmallVec::new();
    let mut common_suffix: Option<u32> = None;

    for state in states {
        let Action::StackShifts(shifts) = table.action(state, token)? else {
            return None;
        };
        let [shift] = shifts.as_slice() else {
            return None;
        };
        if shift.pop != 1 || shift.pushes.len() != 2 {
            return None;
        }

        let prefix = shift.pushes[0];
        let suffix = shift.pushes[1];
        if common_suffix.replace(suffix).is_some_and(|existing| existing != suffix) {
            return None;
        }
        per_state_prefixes.push((state, prefix, true));
    }

    let common_suffix = common_suffix?;
    Some(closure.apply_top_pure_shifts(per_state_prefixes).push(common_suffix))
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
    if let Some(first) = shifts.first()
        && shifts
            .iter()
            .all(|shift| shift.pop == first.pop && shift.pushes.len() == 1)
        && let Some(shifted) = gss.apply_shared_pop_push_single_branches(
            first.pop as usize,
            shifts.iter().map(|shift| &shift.pushes[0]),
        )
    {
        return shifted;
    }

    if let Some(first) = shifts.first()
        && !first.pushes.is_empty()
        && shifts
            .iter()
            .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
        && let Some(shifted) = gss.apply_shared_pop_push_branches(
            first.pop as usize,
            shifts.iter().map(|shift| shift.pushes.as_slice()),
        )
    {
        return shifted;
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

    if let [shift] = shifts
        && let Some(mut stack) = gss.try_virtual_stack()
        && stack.pop(shift.pop as usize) == 0
    {
        for &state in &shift.pushes {
            stack.push(state);
        }
        return stack.into_gss();
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

fn apply_guarded_stack_shifts(gss: ParserGSS, shifts: &[GuardedStackShift]) -> ParserGSS {
    if let Some(stack) = gss.try_virtual_stack() {
        return apply_guarded_stack_shifts_to_vstack(&stack, shifts);
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
    ) {
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

fn apply_guarded_stack_shifts_to_vstack(
    stack: &VirtualStack<u32, TerminalsDisallowed>,
    shifts: &[GuardedStackShift],
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

    for shift in shifts {
        debug_assert!(shift.guards.windows(2).all(|w| w[0].pop <= w[1].pop));
        debug_assert!(shift.guards.iter().all(|guard| guard.pop <= shift.pop));

        let mut dead = false;

        for guard in &shift.guards {
            let Some(state) = state_after_popping(stack, &mut state_after_pop_cache, guard.pop) else {
                dead = true;
                break;
            };
            if guard.states.binary_search(&state).is_err() {
                dead = true;
                break;
            }
        }

        if dead || shift.pop as usize > stack_len {
            continue;
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
                return (AdvancedBranch::Gss(apply_guarded_stack_shifts_to_vstack(&stack, shifts)), true);
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
    let Some(mut stack) = gss.try_virtual_stack() else {
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
                *gss = apply_guarded_stack_shifts_to_vstack(&stack, shifts);
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
                        apply_guarded_stack_shifts_to_vstack(&stack, shifts)
                    } else {
                        apply_guarded_stack_shifts(isolated, shifts)
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
                        apply_guarded_stack_shifts_to_vstack(&stack, shifts)
                    } else {
                        apply_guarded_stack_shifts(isolated, shifts)
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
    let Some(mut stack) = gss.try_virtual_stack() else {
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
                *gss = apply_guarded_stack_shifts_to_vstack(&stack, shifts);
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

pub(crate) fn stack_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    let virtual_stack = stack.try_virtual_stack();
    stack.peek_values().into_iter().any(|state| {
        let Some(action) = table.action(state, token) else {
            return false;
        };
        match action {
            Action::GuardedStackShifts(shifts) => virtual_stack
                .as_ref()
                .is_some_and(|vstack| {
                    shifts
                        .iter()
                        .any(|shift| virtual_stack_may_apply_guarded_shift(vstack, shift))
                })
                || {
                    let isolated = stack.isolate(Some(state));
                    !apply_guarded_stack_shifts(isolated, shifts).is_empty()
                },
            _ => true,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{ParserGSS, advance_stacks};
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::table::testing::build_test_table;
    use crate::compiler::glr::table::{Action, StackShift};

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
}

pub(crate) fn stack_may_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    let virtual_stack = stack.try_virtual_stack();
    stack.peek_values().into_iter().any(|state| {
        table.action.get(state as usize).is_some_and(|actions| {
            actions.iter().any(|(terminal, action)| {
                if !terminals.contains(terminal as usize) {
                    return false;
                }
                match action {
                    Action::GuardedStackShifts(shifts) => virtual_stack
                        .as_ref()
                        .is_some_and(|vstack| {
                            shifts
                                .iter()
                                .any(|shift| virtual_stack_may_apply_guarded_shift(vstack, shift))
                        })
                        || {
                            let isolated = stack.isolate(Some(state));
                            !apply_guarded_stack_shifts(isolated, shifts).is_empty()
                        },
                    _ => true,
                }
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

    has_eof_action
}
