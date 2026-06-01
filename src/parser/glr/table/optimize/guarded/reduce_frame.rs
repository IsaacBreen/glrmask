fn apply_reduce_to_frame(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
) -> Option<ReduceFrameResult> {
    pop_frame(&mut frame, len);

    let mut origin_dependent = false;
    let goto_froms = if let Some(&state) = frame.pushes.last() {
        BTreeSet::from([state])
    } else {
        origin_dependent = true;
        states_at_depth(predecessors, origin_state, frame.pop, states_at_depth_cache)?
    };

    let guard_pop = frame.pop;
    let mut by_target: BTreeMap<(u32, bool), BTreeSet<u32>> = BTreeMap::new();
    let mut missing = 0usize;
    for goto_from in goto_froms {
        let Some((next_target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            missing += 1;
            continue;
        };
        by_target
            .entry((next_target, replace))
            .or_default()
            .insert(goto_from);
    }

    if missing > 0 && by_target.is_empty() {
        return Some(ReduceFrameResult::Dead);
    }

    let needs_guard = missing > 0 || by_target.len() > 1;
    let mut frames = Vec::new();
    for ((target, replace), froms) in by_target {
        let mut next_frame = frame.clone();
        if needs_guard && !add_guard_to_frame(&mut next_frame, guard_pop, froms.into_iter()) {
            continue;
        }
        push_transition_to_frame(&mut next_frame, target, replace);
        frames.push(next_frame);
    }

    if frames.is_empty() {
        Some(ReduceFrameResult::Dead)
    } else {
        frames.sort();
        frames.dedup();
        Some(ReduceFrameResult::Frames {
            frames,
            origin_dependent,
        })
    }
}

fn compose_guarded_shift_with_frame(
    mut frame: StackEffectFrame,
    shift: &GuardedStackShift,
) -> Option<Option<StackEffectFrame>> {
    let pushed_len = frame.pushes.len() as u32;

    for guard in &shift.guards {
        if guard.states.is_empty() {
            return Some(None);
        }

        if guard.pop < pushed_len {
            let idx = (pushed_len - 1 - guard.pop) as usize;
            let known_state = frame.pushes[idx];
            if guard.states.binary_search(&known_state).is_err() {
                return Some(None);
            }
        } else {
            let translated_pop = frame.pop + (guard.pop - pushed_len);
            if !add_guard_to_frame(&mut frame, translated_pop, guard.states.iter().copied()) {
                return Some(None);
            }
        }
    }

    if shift.pop < shift.guards.iter().map(|guard| guard.pop).max().unwrap_or(0) {
        return None;
    }

    pop_frame(&mut frame, shift.pop);
    frame.pushes.extend_from_slice(&shift.pushes);
    Some(Some(frame))
}
