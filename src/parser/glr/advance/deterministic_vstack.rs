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
                    AdvancedBranch::Gss(apply_guarded_stack_shifts_to_vstack(
                        &stack,
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
