#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StackEffectFrame {
    pop: u32,
    pushes: Vec<u32>,
    guards: Vec<StackShiftGuard>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReduceFrameResult {
    Dead,
    Frames {
        frames: Vec<StackEffectFrame>,
        origin_dependent: bool,
    },
}

fn states_at_depth(
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    depth: u32,
    cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
) -> Option<BTreeSet<u32>> {
    if let Some(cached) = cache.get(&(origin_state, depth)) {
        return cached.clone();
    }

    let mut states = BTreeSet::from([origin_state]);
    for _ in 0..depth {
        let mut next = BTreeSet::new();
        for state in states {
            next.extend(predecessors.get(state as usize)?.iter().copied());
        }
        if next.is_empty() {
            cache.insert((origin_state, depth), None);
            return None;
        }
        states = next;
    }

    let result = Some(states);
    cache.insert((origin_state, depth), result.clone());
    result
}

fn normalize_states(mut states: Vec<u32>) -> Vec<u32> {
    states.sort_unstable();
    states.dedup();
    states
}

fn add_guard_to_frame(
    frame: &mut StackEffectFrame,
    pop: u32,
    states: impl IntoIterator<Item = u32>,
) -> bool {
    let states = normalize_states(states.into_iter().collect());
    if states.is_empty() {
        return false;
    }

    if let Some(existing) = frame.guards.iter_mut().find(|guard| guard.pop == pop) {
        let wanted: BTreeSet<u32> = states.into_iter().collect();
        existing.states.retain(|state| wanted.contains(state));
        return !existing.states.is_empty();
    }

    frame.guards.push(StackShiftGuard { pop, states });
    frame.guards.sort_by_key(|guard| guard.pop);
    true
}

fn pop_frame(frame: &mut StackEffectFrame, pop: u32) {
    if pop as usize <= frame.pushes.len() {
        let keep = frame.pushes.len() - pop as usize;
        frame.pushes.truncate(keep);
    } else {
        frame.pop += pop - frame.pushes.len() as u32;
        frame.pushes.clear();
    }
}

fn push_transition_to_frame(frame: &mut StackEffectFrame, target: u32, replace: bool) {
    if replace {
        if let Some(top) = frame.pushes.last_mut() {
            *top = target;
        } else {
            frame.pop += 1;
            frame.pushes.push(target);
        }
    } else {
        frame.pushes.push(target);
    }
}

fn frame_to_guarded_shift(frame: StackEffectFrame) -> GuardedStackShift {
    GuardedStackShift {
        guards: frame.guards,
        pop: frame.pop,
        pushes: frame.pushes,
    }
}

fn stack_effect_action_key(action: &Action) -> StackEffectActionKey {
    match action {
        Action::Shift(target, replace) => StackEffectActionKey::Shift(*target, *replace),
        Action::StackShifts(shifts) => StackEffectActionKey::StackShifts(shifts.clone()),
        Action::GuardedStackShifts(shifts) => {
            StackEffectActionKey::GuardedStackShifts(shifts.clone())
        }
        Action::Reduce(nt, len) => StackEffectActionKey::Reduce(*nt, *len),
        Action::Split { .. } => StackEffectActionKey::Split,
        Action::Accept => StackEffectActionKey::Accept,
    }
}

fn stack_effect_action_tag(action: &Action) -> u8 {
    match action {
        Action::Shift(..) => 0,
        Action::StackShifts(_) => 1,
        Action::GuardedStackShifts(_) => 2,
        Action::Reduce(..) => 3,
        Action::Split { .. } => 4,
        Action::Accept => 5,
    }
}
