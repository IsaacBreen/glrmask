enum CellUpdate {
    Set(Action),
    Remove,
}

fn build_runtime_state_predecessors(
    table: &GLRTable,
    original_num_states: u32,
    constituent_sets: &[BTreeSet<u32>],
) -> Vec<BTreeSet<u32>> {
    let mut predecessors = vec![BTreeSet::new(); table.num_states as usize];

    for src in 0..table.num_states as usize {
        for action in table.action[src].values() {
            match action {
                Action::Shift(dst, false) => {
                    predecessors[*dst as usize].extend(constituent_sets[src].iter().copied());
                }
                Action::Split { shift: Some((dst, false)), .. } => {
                    predecessors[*dst as usize].extend(constituent_sets[src].iter().copied());
                }
                _ => {}
            }
        }
        for &(dst, replace) in table.goto[src].values() {
            if !replace {
                predecessors[dst as usize].extend(constituent_sets[src].iter().copied());
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for src in 0..original_num_states as usize {
            let src_preds = predecessors[src].clone();
            for action in table.action[src].values() {
                match action {
                    Action::Shift(dst, true) => {
                        let before = predecessors[*dst as usize].len();
                        predecessors[*dst as usize].extend(src_preds.iter().copied());
                        changed |= predecessors[*dst as usize].len() != before;
                    }
                    Action::Split { shift: Some((dst, true)), .. } => {
                        let before = predecessors[*dst as usize].len();
                        predecessors[*dst as usize].extend(src_preds.iter().copied());
                        changed |= predecessors[*dst as usize].len() != before;
                    }
                    _ => {}
                }
            }
            for &(dst, replace) in table.goto[src].values() {
                if replace {
                    let before = predecessors[dst as usize].len();
                    predecessors[dst as usize].extend(src_preds.iter().copied());
                    changed |= predecessors[dst as usize].len() != before;
                }
            }
        }
    }

    predecessors
}

fn subset_key(subset: &BTreeSet<u32>) -> Vec<u32> {
    subset.iter().copied().collect()
}

fn union_state_subsets(
    states: impl IntoIterator<Item = u32>,
    constituent_sets: &[BTreeSet<u32>],
) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for state in states {
        out.extend(constituent_sets[state as usize].iter().copied());
    }
    out
}

fn merge_shift_into_pending(
    pending: &mut PendingAction,
    target: u32,
    replace: bool,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<(), ()> {
    match pending.shift {
        None => {
            pending.shift = Some((target, replace));
            Ok(())
        }
        Some((existing_target, existing_replace)) => {
            if existing_target == target {
                return if existing_replace == replace { Ok(()) } else { Err(()) };
            }
            if existing_replace != replace {
                return Err(());
            }
            let merged_subset = union_state_subsets([existing_target, target], constituent_sets);
            let merged_target = ensure_subset_state(
                table,
                &merged_subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
            )?;
            pending.shift = Some((merged_target, replace));
            Ok(())
        }
    }
}

fn merge_action_into_pending(
    pending: &mut PendingAction,
    action: &Action,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<(), ()> {
    match action {
        Action::Shift(target, replace) => merge_shift_into_pending(
            pending,
            *target,
            *replace,
            table,
            constituent_sets,
            subset_to_state,
            failed_subsets,
        ),
        Action::StackShifts(_) => Err(()),
        Action::GuardedStackShifts(_) => Err(()),
        Action::Reduce(nt, len) => {
            pending.push_reduce(*nt, *len);
            Ok(())
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                merge_shift_into_pending(
                    pending,
                    *target,
                    *replace,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                )?;
            }
            for &(nt, len) in reduces {
                pending.push_reduce(nt, len);
            }
            if *accept {
                pending.push_accept();
            }
            Ok(())
        }
        Action::Accept => {
            pending.push_accept();
            Ok(())
        }
    }
}

fn build_merged_action_row(
    table: &mut GLRTable,
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<ActionRow, ()> {
    let mut terminals = BTreeSet::new();
    for &state in subset {
        for tid in table.action[state as usize].keys() {
            terminals.insert(tid);
        }
    }

    let mut row = ActionRow::default();
    for tid in terminals {
        let mut pending = PendingAction::default();
        for &state in subset {
            if let Some(action) = table.action[state as usize].get(&tid).cloned() {
                merge_action_into_pending(
                    &mut pending,
                    &action,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                )?;
            }
        }
        if let Some(action) = pending.maybe_finish() {
            row.insert(tid, action);
        }
    }

    Ok(row)
}

fn build_merged_goto_row(
    table: &mut GLRTable,
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<GotoRow, ()> {
    let mut nts = BTreeSet::new();
    for &state in subset {
        for &nt in table.goto[state as usize].keys() {
            nts.insert(nt);
        }
    }

    let mut row = GotoRow::default();
    for nt in nts {
        let mut replace: Option<bool> = None;
        let mut target_subset = BTreeSet::new();
        let mut saw_target = false;

        for &state in subset {
            if let Some(&(target, is_replace)) = table.goto[state as usize].get(&nt) {
                saw_target = true;
                match replace {
                    None => replace = Some(is_replace),
                    Some(existing) if existing == is_replace => {}
                    Some(_) => return Err(()),
                }
                target_subset.extend(constituent_sets[target as usize].iter().copied());
            }
        }

        if !saw_target {
            continue;
        }

        let merged_target = ensure_subset_state(
            table,
            &target_subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
        )?;
        row.insert(nt, (merged_target, replace.unwrap()));
    }

    Ok(row)
}

fn union_advance_rows(table: &GLRTable, subset: &BTreeSet<u32>) -> BitSet {
    let mut out = BitSet::new(table.num_terminals as usize + 1);
    if table.advance.len() == table.num_states as usize {
        for &state in subset {
            out.union_with(&table.advance[state as usize]);
        }
    } else {
        for &state in subset {
            for terminal in table.action[state as usize].keys() {
                let bit = if terminal == EOF {
                    table.num_terminals as usize
                } else if terminal < table.num_terminals {
                    terminal as usize
                } else {
                    continue;
                };
                out.set(bit);
            }
        }
    }
    out
}

fn ensure_subset_state(
    table: &mut GLRTable,
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<u32, ()> {
    debug_assert!(!subset.is_empty());
    if subset.len() == 1 {
        return Ok(*subset.iter().next().unwrap());
    }

    let key = subset_key(subset);
    if let Some(&state) = subset_to_state.get(&key) {
        return Ok(state);
    }
    if failed_subsets.contains(&key) {
        return Err(());
    }

    let had_advance_rows = table.advance.len() == table.num_states as usize;
    let advance_row = had_advance_rows.then(|| union_advance_rows(table, subset));

    let state = table.num_states;
    table.num_states += 1;
    table.action.push(ActionRow::default());
    table.goto.push(GotoRow::default());
    if let Some(advance_row) = advance_row {
        table.advance.push(advance_row);
    }
    constituent_sets.push(subset.clone());
    subset_to_state.insert(key.clone(), state);

    let built = (|| {
        let action_row = build_merged_action_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
        )?;
        let goto_row = build_merged_goto_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
        )?;
        Ok::<_, ()>((action_row, goto_row))
    })();

    match built {
        Ok((action_row, goto_row)) => {
            table.action[state as usize] = action_row;
            table.goto[state as usize] = goto_row;
            Ok(state)
        }
        Err(()) => {
            subset_to_state.remove(&key);
            failed_subsets.insert(key);
            table.action.pop();
            table.goto.pop();
            if had_advance_rows {
                table.advance.pop();
            }
            table.num_states -= 1;
            constituent_sets.pop();
            Err(())
        }
    }
}

fn refresh_merged_states_depending_on(
    table: &mut GLRTable,
    original_num_states: u32,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    dirty_original_states: &BTreeSet<u32>,
) {
    let mut state = original_num_states as usize;
    while state < table.num_states as usize {
        // Merged-state rows depend only on their flattened original
        // constituents, so only subsets intersecting changed original rows
        // need to be rebuilt.
        if constituent_sets[state].is_disjoint(dirty_original_states) {
            state += 1;
            continue;
        }

        let subset = constituent_sets[state].clone();
        let rebuilt = (|| {
            let action_row = build_merged_action_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
            )?;
            let goto_row = build_merged_goto_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
            )?;
            Ok::<_, ()>((action_row, goto_row))
        })();

        if let Ok((action_row, goto_row)) = rebuilt {
            table.action[state] = action_row;
            table.goto[state] = goto_row;
            if table.advance.len() == table.num_states as usize {
                table.advance[state] = union_advance_rows(table, &subset);
            }
        }

        state += 1;
    }
}

fn unit_reduce_destination(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    lhs: NonterminalID,
) -> Option<u32> {
    let preds = &predecessors[state as usize];
    assert!(!preds.is_empty());

    let relevant_preds: Vec<u32> = preds
        .iter()
        .copied()
        .filter(|&pred| table.goto[pred as usize].contains_key(&lhs))
        .collect();
    if relevant_preds.is_empty() {
        return None;
    }

    let mut reduce_dst: Option<u32> = None;
    for pred in relevant_preds {
        let (dst, is_replace) = table.goto[pred as usize][&lhs];
        if is_replace {
            return None;
        }
        if table.goto[dst as usize] != table.goto[state as usize] {
            return None;
        }
        match reduce_dst {
            None => reduce_dst = Some(dst),
            Some(existing) if existing == dst => {}
            Some(_) => return None,
        }
    }

    reduce_dst
}

