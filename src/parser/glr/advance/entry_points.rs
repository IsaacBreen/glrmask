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
