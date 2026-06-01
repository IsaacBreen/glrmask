fn stack_effect_action(table: &GLRTable, mut effects: Vec<GuardedStackShift>) -> Option<Action> {
    normalize_guarded_effects(&mut effects);
    if effects.is_empty() {
        return None;
    }
    if effects.iter().all(|effect| effect.guards.is_empty()) {
        let mut shifts: Vec<_> = effects
            .into_iter()
            .map(|effect| StackShift {
                pop: effect.pop,
                pushes: effect.pushes,
            })
            .collect();
        if stack_shift_predecessor_canonicalization_enabled() {
            canonicalize_stack_shift_predecessors_by_goto_superset(table, &mut shifts);
        }
        return stack_shift_action(shifts);
    }
    // Opt-in diagnostic knob only. Do not use this to hide correctness or
    // compile-time bugs in guarded stack-effect lowering.
    if max_guarded_stack_effects().is_some_and(|limit| effects.len() > limit) {
        return None;
    }
    Some(Action::GuardedStackShifts(effects))
}

fn try_inline_action_to_stack_shifts(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    stack_effect_memo: &mut FxHashMap<StackEffectKey, Option<StackEffectResult>>,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
) -> Option<Action> {
    let has_reductions = match action {
        Action::Reduce(..) => true,
        Action::Split {
            reduces,
            accept: false,
            ..
        } => !reduces.is_empty(),
        _ => false,
    };
    if !has_reductions {
        return None;
    }

    let result = stack_effects_for_action(
        table,
        predecessors,
        state,
        tid,
        state,
        action,
        StackEffectFrame {
            pop: 0,
            pushes: Vec::new(),
            guards: Vec::new(),
        },
        states_at_depth_cache,
        &mut FxHashSet::default(),
        stack_effect_memo,
    )?;
    let effects = result.effects;
    if result.origin_dependent
        && matches!(action, Action::Reduce(_, 1))
        && !effects.is_empty()
        && effects.iter().all(|effect| effect.pushes.len() == 1)
    {
        return None;
    }
    if effects.is_empty() {
        return None;
    }
    stack_effect_action(table, effects)
}
