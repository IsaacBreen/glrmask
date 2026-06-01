fn advance_nondeterministically(
    table: &GLRTable,
    mut closure: ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    let mut shifted = ParserGSS::empty();

    loop {
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
                        apply_guarded_stack_shifts_to_vstack(
                            &stack,
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
