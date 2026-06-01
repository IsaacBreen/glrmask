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
