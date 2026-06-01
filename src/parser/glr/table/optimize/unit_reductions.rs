fn try_inline_unit_reductions_for_cell(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    stack_effect_memo: &mut FxHashMap<StackEffectKey, Option<StackEffectResult>>,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<Option<CellUpdate>, ()> {
    if let Some(action) = try_inline_action_to_stack_shifts(
        table,
        predecessors,
        state,
        tid,
        action,
        stack_effect_memo,
        states_at_depth_cache,
    ) {
        return Ok(Some(CellUpdate::Set(action)));
    }

    match action {
        Action::Split {
            shift: Some(_),
            accept: false,
            ..
        }
        | Action::Shift(_, _) => {}
        _ => return Ok(None),
    }

    let mut visiting = BTreeSet::new();
    try_inline_unit_reductions_for_cell_inner(
        table,
        predecessors,
        state,
        tid,
        action,
        constituent_sets,
        subset_to_state,
        failed_subsets,
        &mut visiting,
    )
}

fn try_inline_unit_reductions_for_cell_inner(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    visiting: &mut BTreeSet<(u32, TerminalID)>,
) -> Result<Option<CellUpdate>, ()> {
    if !visiting.insert((state, tid)) {
        return Ok(None);
    }

    let mut pending = PendingAction::default();
    let mut reduces: Vec<(NonterminalID, u32)> = Vec::new();

    match action {
        Action::Shift(target, replace) => pending.push_shift(*target, *replace),
        Action::StackShifts(_) => return Ok(None),
        Action::GuardedStackShifts(_) => return Ok(None),
        Action::Reduce(nt, len) => reduces.push((*nt, *len)),
        Action::Split {
            shift,
            reduces: action_reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                pending.push_shift(*target, *replace);
            }
            reduces.extend(action_reduces.iter().copied());
            if *accept {
                pending.push_accept();
            }
        }
        Action::Accept => pending.push_accept(),
    }

    let mut changed = false;
    for (lhs, pop_len) in reduces {
        if pop_len != 1 {
            pending.push_reduce(lhs, pop_len);
            continue;
        }

        let Some(reduce_dst) = unit_reduce_destination(table, predecessors, state, lhs) else {
            pending.push_reduce(lhs, pop_len);
            continue;
        };

        match table.action[reduce_dst as usize].get(&tid).cloned() {
            None => {
                pending.push_reduce(lhs, pop_len);
            }
            Some(inline_action) => {
                let resolved_inline = match try_inline_unit_reductions_for_cell_inner(
                    table,
                    predecessors,
                    reduce_dst,
                    tid,
                    &inline_action,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    visiting,
                )? {
                    Some(CellUpdate::Set(action)) => Some(action),
                    Some(CellUpdate::Remove) => None,
                    None => Some(inline_action),
                };

                let Some(resolved_inline) = resolved_inline else {
                    changed = true;
                    continue;
                };

                merge_action_into_pending(
                    &mut pending,
                    &resolved_inline,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                )?;
                changed = true;
            }
        }
    }

    let result = if !changed {
        Ok(None)
    } else {
        Ok(match pending.maybe_finish() {
            Some(action) => Some(CellUpdate::Set(action)),
            None => Some(CellUpdate::Remove),
        })
    };
    visiting.remove(&(state, tid));
    result
}

fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {
    match action {
        Action::Shift(target, replace) => Action::Shift(mapping[*target as usize], *replace),
        Action::StackShifts(shifts) => {
            let mut remapped = shifts
                .iter()
                .map(|shift| StackShift {
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| mapping[state as usize]).collect(),
                })
                .collect();
            normalize_stack_shifts(&mut remapped);
            Action::StackShifts(remapped)
        }
        Action::GuardedStackShifts(shifts) => Action::GuardedStackShifts(
            shifts
                .iter()
                .map(|shift| GuardedStackShift {
                    guards: shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states: Vec<u32> = guard
                                .states
                                .iter()
                                .map(|&state| mapping[state as usize])
                                .collect();
                            states.sort_unstable();
                            states.dedup();
                            StackShiftGuard {
                                pop: guard.pop,
                                states,
                            }
                        })
                        .collect(),
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| mapping[state as usize]).collect(),
                })
                .collect(),
        ),
        Action::Reduce(nt, len) => Action::Reduce(*nt, *len),
        Action::Split {
            shift,
            reduces,
            accept,
        } => Action::Split {
            shift: shift.map(|(target, replace)| (mapping[target as usize], replace)),
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => Action::Accept,
    }
}

