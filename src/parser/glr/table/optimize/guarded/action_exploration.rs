fn stack_effects_for_action(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    tid: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    memo: &mut FxHashMap<StackEffectKey, Option<StackEffectResult>>,
) -> Option<StackEffectResult> {
    let memo_key = StackEffectKey {
        origin_state,
        state,
        tid,
        action: stack_effect_action_key(action),
        frame: frame.clone(),
    };
    if let Some(cached) = memo.get(&memo_key) {
        return cached.clone();
    }

    let key = StackEffectVisitKey {
        state,
        tid,
        action_tag: stack_effect_action_tag(action),
        frame: frame.clone(),
    };
    if !visiting.insert(key.clone()) {
        return None;
    }

    let result = (|| {
        let mut out = Vec::new();
        let mut origin_dependent = false;
        match action {
            Action::Shift(target, replace) => {
                let mut frame = frame;
                let effective_replace = *replace && !table.forwarded_shifts.contains(&(state, tid));
                push_transition_to_frame(&mut frame, *target, effective_replace);
                out.push(frame_to_guarded_shift(frame));
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    let mut frame = frame.clone();
                    pop_frame(&mut frame, shift.pop);
                    frame.pushes.extend_from_slice(&shift.pushes);
                    out.push(frame_to_guarded_shift(frame));
                }
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    match compose_guarded_shift_with_frame(frame.clone(), shift)? {
                        None => {}
                        Some(frame) => out.push(frame_to_guarded_shift(frame)),
                    }
                }
            }
            Action::Reduce(nt, len) => {
                let frames = match apply_reduce_to_frame(
                    table,
                    predecessors,
                    origin_state,
                    frame,
                    *nt,
                    *len,
                    states_at_depth_cache,
                )? {
                    ReduceFrameResult::Dead => {
                        return Some(StackEffectResult {
                            effects: Vec::new(),
                            origin_dependent,
                        })
                    }
                    ReduceFrameResult::Frames {
                        frames,
                        origin_dependent: reduce_origin_dependent,
                    } => {
                        origin_dependent |= reduce_origin_dependent;
                        frames
                    }
                };
                for frame in frames {
                    let Some(&next_state) = frame.pushes.last() else {
                        continue;
                    };
                    let Some(next) = table.action[next_state as usize].get(&tid) else {
                        continue;
                    };
                    let next_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        next_state,
                        next,
                        frame,
                        states_at_depth_cache,
                        visiting,
                        memo,
                    )?;
                    origin_dependent |= next_result.origin_dependent;
                    out.extend(next_result.effects);
                }
            }
            Action::Split { shift, reduces, accept } => {
                if *accept {
                    return None;
                }
                if let Some((target, replace)) = shift {
                    let shift_action = Action::Shift(*target, *replace);
                    let shift_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        state,
                        &shift_action,
                        frame.clone(),
                        states_at_depth_cache,
                        visiting,
                        memo,
                    )?;
                    origin_dependent |= shift_result.origin_dependent;
                    out.extend(shift_result.effects);
                }
                for &(nt, len) in reduces {
                    let reduce_action = Action::Reduce(nt, len);
                    let reduce_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        state,
                        &reduce_action,
                        frame.clone(),
                        states_at_depth_cache,
                        visiting,
                        memo,
                    )?;
                    origin_dependent |= reduce_result.origin_dependent;
                    out.extend(reduce_result.effects);
                }
            }
            Action::Accept => return None,
        }

        out.sort();
        out.dedup();
        Some(StackEffectResult {
            effects: out,
            origin_dependent,
        })
    })();

    visiting.remove(&key);
    memo.insert(memo_key, result.clone());
    result
}

fn normalize_guarded_effects(effects: &mut Vec<GuardedStackShift>) {
    for effect in effects.iter_mut() {
        for guard in effect.guards.iter_mut() {
            guard.states.sort_unstable();
            guard.states.dedup();
        }
        effect.guards.retain(|guard| !guard.states.is_empty());
        effect.guards.sort_by_key(|guard| guard.pop);
        effect.guards.dedup();
    }
    effects.retain(|effect| !effect.pushes.is_empty());
    effects.sort();
    effects.dedup();
}
