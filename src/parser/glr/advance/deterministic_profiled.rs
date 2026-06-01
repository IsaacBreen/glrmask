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
                *gss = apply_guarded_stack_shifts_to_vstack(
                    &stack,
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
