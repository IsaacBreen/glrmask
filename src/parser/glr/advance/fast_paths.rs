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

fn try_advance_pop1_reduce_plus_stackshift_wave(
    table: &GLRTable,
    closure: &ParserGSS,
    token: TerminalID,
) -> Option<ParserGSS> {
    let states = closure.peek_values();
    if !(2..=3).contains(&states.len()) {
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

    if let Some(stack) = gss.try_virtual_stack()
        && let Some(first) = shifts.first()
        && !first.pushes.is_empty()
        && shifts
            .iter()
            .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
        && let Some(shifted) = stack.into_gss_after_popping_and_pushing_branches(
            first.pop as usize,
            shifts.iter().map(|shift| shift.pushes.as_slice()),
        )
    {
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

pub(crate) fn apply_guarded_stack_shifts_fast(
    gss: &ParserGSS,
    shifts: &[GuardedStackShift],
    index: Option<&GuardedShiftCellIndex>,
) -> Option<ParserGSS> {
    if let Some(stack) = gss.try_virtual_stack() {
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
