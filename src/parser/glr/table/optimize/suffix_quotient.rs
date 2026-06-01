struct SuffixQuotient {
    suffix_to_state: FxHashMap<Vec<Vec<u32>>, u32>,
    failed_suffixes: FxHashSet<Vec<Vec<u32>>>,
    max_states: usize,
    max_alts: usize,
    max_width: usize,
    created_states: usize,
}

impl SuffixQuotient {
    fn new() -> Self {
        Self {
            suffix_to_state: FxHashMap::default(),
            failed_suffixes: FxHashSet::default(),
            // Do not lower this default to mask correctness or crash bugs.
            // If a schema fails only above this cap, fix the quotient/table
            // invariant that fails at scale instead of hiding the failure.
            max_states: table_options_from_env().recognizer_suffix_quotient_max_states,
            max_alts: table_options_from_env().recognizer_suffix_quotient_max_alts,
            max_width: table_options_from_env().recognizer_suffix_quotient_max_width,
            created_states: 0,
        }
    }

    fn normalize_action(&mut self, table: &mut GLRTable, action: Action) -> Option<Action> {
        match action {
            Action::StackShifts(shifts) => {
                let mut effects = shifts
                    .into_iter()
                    .map(|shift| GuardedStackShift {
                        guards: Vec::new(),
                        pop: shift.pop,
                        pushes: shift.pushes,
                    })
                    .collect::<Vec<_>>();
                let normalized = self.quotient_effect_groups(table, &mut effects).ok()?;
                if Self::action_has_multi_stack_shifts(&normalized) {
                    return None;
                }
                Some(normalized)
            }
            Action::GuardedStackShifts(mut effects) => {
                let normalized = self.quotient_effect_groups(table, &mut effects).ok()?;
                if Self::action_has_multi_stack_shifts(&normalized) {
                    return None;
                }
                Some(normalized)
            }
            Action::Split {
                shift,
                reduces,
                accept,
            } if reduces.is_empty() && !accept => {
                shift.map(|(target, replace)| Action::Shift(target, replace))
            }
            _ => None,
        }
    }

    fn quotient_effect_groups(
        &mut self,
        table: &mut GLRTable,
        effects: &mut Vec<GuardedStackShift>,
    ) -> Result<Action, ()> {
        normalize_guarded_effects_for_suffix_quotient(effects);
        if effects.is_empty() {
            return Err(());
        }

        let mut groups: BTreeMap<(Vec<StackShiftGuard>, u32), Vec<Vec<u32>>> = BTreeMap::new();
        for effect in effects.iter() {
            groups
                .entry((effect.guards.clone(), effect.pop))
                .or_default()
                .push(effect.pushes.clone());
        }

        let mut out = Vec::new();
        for ((guards, pop), mut suffixes) in groups {
            normalize_suffixes(&mut suffixes);
            if suffixes.len() > 1
                && suffixes.iter().all(|suffix| !suffix.is_empty())
                && let Ok(target) = self.ensure_suffix_state(table, suffixes.clone())
            {
                out.push(GuardedStackShift {
                    guards,
                    pop,
                    pushes: vec![target],
                });
            } else {
                for pushes in suffixes {
                    out.push(GuardedStackShift {
                        guards: guards.clone(),
                        pop,
                        pushes,
                    });
                }
            }
        }

        normalize_guarded_effects_for_suffix_quotient(&mut out);
        if out.iter().all(|effect| effect.guards.is_empty()) {
            let shifts = out
                .into_iter()
                .map(|effect| StackShift {
                    pop: effect.pop,
                    pushes: effect.pushes,
                })
                .collect();
            return stack_shift_action(shifts).ok_or(());
        }
        Ok(Action::GuardedStackShifts(out))
    }

    fn ensure_suffix_state(
        &mut self,
        table: &mut GLRTable,
        mut suffixes: Vec<Vec<u32>>,
    ) -> Result<u32, ()> {
        normalize_suffixes(&mut suffixes);
        if suffixes.is_empty() || suffixes.iter().any(|suffix| suffix.is_empty()) {
            return Err(());
        }
        if suffixes.len() > self.max_alts || suffixes.iter().any(|suffix| suffix.len() > self.max_width) {
            return Err(());
        }
        if suffixes.len() == 1 && suffixes[0].len() == 1 {
            return Ok(suffixes[0][0]);
        }
        if let Some(&state) = self.suffix_to_state.get(&suffixes) {
            return Ok(state);
        }
        if self.failed_suffixes.contains(&suffixes) || self.created_states >= self.max_states {
            return Err(());
        }

        let rollback_state = table.num_states;
        let rollback_created_states = self.created_states;
        let had_advance_rows = table.advance.len() == table.num_states as usize;
        let state = rollback_state;
        table.num_states += 1;
        table.action.push(ActionRow::default());
        table.goto.push(GotoRow::default());
        if had_advance_rows {
            table
                .advance
                .push(super::action_presence_row(&ActionRow::default(), table.num_terminals));
        }
        self.created_states += 1;
        self.suffix_to_state.insert(suffixes.clone(), state);
        let built = (|| {
            let action = self.build_suffix_action_row(table, &suffixes)?;
            let goto = self.build_suffix_goto_row(table, &suffixes)?;
            Ok::<_, ()>((action, goto))
        })();

        match built {
            Ok((action, goto)) => {
                if Self::action_row_has_multi_stack_shifts(&action) {
                    self.suffix_to_state.retain(|_, mapped_state| *mapped_state < rollback_state);
                    table.action.truncate(rollback_state as usize);
                    table.goto.truncate(rollback_state as usize);
                    if had_advance_rows {
                        table.advance.truncate(rollback_state as usize);
                    }
                    table.num_states = rollback_state;
                    self.created_states = rollback_created_states;
                    self.failed_suffixes.insert(suffixes);
                    return Err(());
                }
                table.action[state as usize] = action;
                table.goto[state as usize] = goto;
                if had_advance_rows {
                    table.advance[state as usize] =
                        super::action_presence_row(&table.action[state as usize], table.num_terminals);
                }
                Ok(state)
            }
            Err(()) => {
                self.suffix_to_state.retain(|_, mapped_state| *mapped_state < rollback_state);
                table.action.truncate(rollback_state as usize);
                table.goto.truncate(rollback_state as usize);
                if had_advance_rows {
                    table.advance.truncate(rollback_state as usize);
                }
                table.num_states = rollback_state;
                self.created_states = rollback_created_states;
                self.failed_suffixes.insert(suffixes);
                Err(())
            }
        }
    }

    fn build_suffix_action_row(
        &mut self,
        table: &mut GLRTable,
        suffixes: &[Vec<u32>],
    ) -> Result<ActionRow, ()> {
        let mut terminals = BTreeSet::new();
        for suffix in suffixes {
            let top = *suffix.last().ok_or(())?;
            for terminal in table.action[top as usize].keys() {
                terminals.insert(terminal);
            }
        }

        let mut row = ActionRow::default();
        for terminal in terminals {
            let mut effects = Vec::new();
            let mut accepts = 0usize;
            for suffix in suffixes {
                let top = *suffix.last().ok_or(())?;
                let Some(action) = table.action[top as usize].get(&terminal).cloned() else {
                    continue;
                };
                self.collect_effects_for_suffix_action(suffix, &action, &mut effects, &mut accepts)?;
            }

            if accepts > 0 {
                if !effects.is_empty() {
                    return Err(());
                }
                row.insert(terminal, Action::Accept);
                continue;
            }

            if effects.is_empty() {
                continue;
            }
            let action = self.quotient_effect_groups(table, &mut effects)?;
            row.insert(terminal, action);
        }
        Ok(row)
    }

    fn build_suffix_goto_row(
        &mut self,
        table: &mut GLRTable,
        suffixes: &[Vec<u32>],
    ) -> Result<GotoRow, ()> {
        let mut nts = BTreeSet::new();
        for suffix in suffixes {
            let top = *suffix.last().ok_or(())?;
            for &nt in table.goto[top as usize].keys() {
                nts.insert(nt);
            }
        }

        let mut row = GotoRow::default();
        for nt in nts {
            let mut result_suffixes = Vec::new();
            for suffix in suffixes {
                let top = *suffix.last().ok_or(())?;
                let Some(&(target, replace)) = table.goto[top as usize].get(&nt) else {
                    continue;
                };
                result_suffixes.push(apply_goto_to_suffix(suffix, target, replace));
            }
            normalize_suffixes(&mut result_suffixes);
            if result_suffixes.is_empty() {
                continue;
            }
            let target = self.ensure_suffix_target(table, result_suffixes)?;
            row.insert(nt, (target, true));
        }
        Ok(row)
    }


fn action_row_has_multi_stack_shifts(row: &ActionRow) -> bool {
    row.values().any(|action| matches!(action, Action::StackShifts(shifts) if shifts.len() > 1))
}

fn action_has_multi_stack_shifts(action: &Action) -> bool {
    matches!(action, Action::StackShifts(shifts) if shifts.len() > 1)
}
    fn ensure_suffix_target(
        &mut self,
        table: &mut GLRTable,
        mut suffixes: Vec<Vec<u32>>,
    ) -> Result<u32, ()> {
        normalize_suffixes(&mut suffixes);
        match suffixes.as_slice() {
            [only] if only.len() == 1 => Ok(only[0]),
            _ => self.ensure_suffix_state(table, suffixes),
        }
    }

    fn collect_effects_for_suffix_action(
        &mut self,
        suffix: &[u32],
        action: &Action,
        effects: &mut Vec<GuardedStackShift>,
        accepts: &mut usize,
    ) -> Result<(), ()> {
        match action {
            Action::Shift(target, replace) => {
                effects.push(unguarded_suffix_effect(
                    suffix,
                    if *replace { 1 } else { 0 },
                    &[*target],
                )?);
                Ok(())
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    effects.push(unguarded_suffix_effect(suffix, shift.pop, &shift.pushes)?);
                }
                Ok(())
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    if let Some(effect) = guarded_suffix_effect(suffix, shift)? {
                        effects.push(effect);
                    }
                }
                Ok(())
            }
            Action::Split {
                shift,
                reduces,
                accept,
            } => {
                if !reduces.is_empty() {
                    return Err(());
                }
                if *accept {
                    *accepts += 1;
                }
                if let Some((target, replace)) = shift {
                    effects.push(unguarded_suffix_effect(
                        suffix,
                        if *replace { 1 } else { 0 },
                        &[*target],
                    )?);
                }
                Ok(())
            }
            Action::Accept => {
                *accepts += 1;
                Ok(())
            }
            Action::Reduce(..) => Err(()),
        }
    }
}

fn normalize_suffixes(suffixes: &mut Vec<Vec<u32>>) {
    suffixes.sort();
    suffixes.dedup();
}

fn normalize_guarded_effects_for_suffix_quotient(effects: &mut Vec<GuardedStackShift>) {
    for effect in effects.iter_mut() {
        for guard in &mut effect.guards {
            guard.states.sort_unstable();
            guard.states.dedup();
        }
        effect.guards.retain(|guard| !guard.states.is_empty());
        effect.guards.sort_by_key(|guard| guard.pop);
        effect.guards.dedup();
    }
    effects.sort();
    effects.dedup();
}

fn apply_goto_to_suffix(suffix: &[u32], target: u32, replace: bool) -> Vec<u32> {
    let mut out = suffix.to_vec();
    if replace {
        if let Some(top) = out.last_mut() {
            *top = target;
        } else {
            out.push(target);
        }
    } else {
        out.push(target);
    }
    out
}

fn unguarded_suffix_effect(
    suffix: &[u32],
    pop: u32,
    pushes: &[u32],
) -> Result<GuardedStackShift, ()> {
    suffix_effect(suffix, Vec::new(), pop, pushes)
}

fn guarded_suffix_effect(
    suffix: &[u32],
    shift: &GuardedStackShift,
) -> Result<Option<GuardedStackShift>, ()> {
    let suffix_len = suffix.len() as u32;
    let mut translated_guards = Vec::new();

    for guard in &shift.guards {
        if guard.pop < suffix_len {
            let index = suffix.len() - 1 - guard.pop as usize;
            if guard.states.binary_search(&suffix[index]).is_err() {
                return Ok(None);
            }
        } else {
            translated_guards.push(StackShiftGuard {
                pop: 1 + (guard.pop - suffix_len),
                states: guard.states.clone(),
            });
        }
    }

    let effect = suffix_effect(suffix, translated_guards, shift.pop, &shift.pushes)?;
    Ok(Some(effect))
}

fn suffix_effect(
    suffix: &[u32],
    mut guards: Vec<StackShiftGuard>,
    pop: u32,
    pushes: &[u32],
) -> Result<GuardedStackShift, ()> {
    if suffix.is_empty() {
        return Err(());
    }
    let suffix_len = suffix.len() as u32;
    let (macro_pop, macro_pushes) = if pop <= suffix_len {
        let keep = suffix.len() - pop as usize;
        let mut out = suffix[..keep].to_vec();
        out.extend_from_slice(pushes);
        (1, out)
    } else {
        (1 + (pop - suffix_len), pushes.to_vec())
    };

    guards.sort_by_key(|guard| guard.pop);
    guards.dedup();
    debug_assert!(guards.iter().all(|guard| guard.pop <= macro_pop));

    Ok(GuardedStackShift {
        guards,
        pop: macro_pop,
        pushes: macro_pushes,
    })
}

fn row_fingerprint(
    action_row: &ActionRow,
    goto_row: &GotoRow,
    advance_row: Option<&BitSet>,
) -> u64 {
    let mut hasher = FxHasher::default();

    let mut action = action_row.iter().collect::<Vec<_>>();
    action.sort_unstable_by_key(|(terminal, _)| *terminal);
    action.len().hash(&mut hasher);
    for (terminal, action) in action {
        terminal.hash(&mut hasher);
        action.hash(&mut hasher);
    }

    let mut goto = goto_row.iter().collect::<Vec<_>>();
    goto.sort_unstable_by_key(|(nonterminal, _)| **nonterminal);
    goto.len().hash(&mut hasher);
    for (nonterminal, target) in goto {
        nonterminal.hash(&mut hasher);
        target.hash(&mut hasher);
    }

    if let Some(advance_row) = advance_row {
        advance_row.hash(&mut hasher);
    }

    hasher.finish()
}

fn rows_equal(
    action_a: &ActionRow,
    goto_a: &GotoRow,
    advance_a: Option<&BitSet>,
    action_b: &ActionRow,
    goto_b: &GotoRow,
    advance_b: Option<&BitSet>,
) -> bool {
    advance_a == advance_b
        && action_a.len() == action_b.len()
        && goto_a.len() == goto_b.len()
        && action_a
            .iter()
            .all(|(terminal, action)| action_b.get(&terminal) == Some(action))
        && goto_a
            .iter()
            .all(|(&nonterminal, target)| goto_b.get(&nonterminal) == Some(target))
}

fn push_reachable_state(state: u32, reachable: &mut [bool], stack: &mut Vec<u32>) {
    let Some(slot) = reachable.get_mut(state as usize) else {
        return;
    };
    if !*slot {
        *slot = true;
        stack.push(state);
    }
}

fn push_action_targets(action: &Action, reachable: &mut [bool], stack: &mut Vec<u32>) {
    match action {
        Action::Shift(target, _) => push_reachable_state(*target, reachable, stack),
        Action::StackShifts(shifts) => {
            for shift in shifts {
                for &state in &shift.pushes {
                    push_reachable_state(state, reachable, stack);
                }
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                for guard in &shift.guards {
                    for &state in &guard.states {
                        push_reachable_state(state, reachable, stack);
                    }
                }
                for &state in &shift.pushes {
                    push_reachable_state(state, reachable, stack);
                }
            }
        }
        Action::Reduce(_, _) | Action::Accept => {}
        Action::Split { shift, .. } => {
            if let Some((target, _)) = shift {
                push_reachable_state(*target, reachable, stack);
            }
        }
    }
}

