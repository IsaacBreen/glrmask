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
