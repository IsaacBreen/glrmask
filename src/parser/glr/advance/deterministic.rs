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
                *gss = apply_guarded_stack_shifts_to_vstack(
                    &stack,
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
/// Ordinary actions (for example shifts and reduces) are applicable from the top
/// state/action row. In particular, LR(1) reduce lookaheads are precise: if the
/// row has a reduce action for this terminal, that reduce is a valid parser
/// transition for the lookahead under the table invariants; it does not require
/// an additional lower-stack guard check here. `GuardedStackShifts` also have
/// lower-stack predicates, so they must evaluate their guards against the current
/// GSS before this predicate can return `true`.
///
