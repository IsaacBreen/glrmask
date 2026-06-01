fn normalize_stack_shifts(shifts: &mut Vec<StackShift>) {
    shifts.sort_by(|a, b| a.pop.cmp(&b.pop).then_with(|| a.pushes.cmp(&b.pushes)));
    shifts.dedup();
}

fn canonicalize_stack_shift_predecessors_by_goto_superset(table: &GLRTable, shifts: &mut [StackShift]) {
    for idx in 0..shifts.len() {
        if shifts[idx].pushes.len() < 2 {
            continue;
        }

        for pos in 0..shifts[idx].pushes.len() - 1 {
            for rep_idx in 0..idx {
                if shifts[idx].pop != shifts[rep_idx].pop
                    || shifts[idx].pushes.len() != shifts[rep_idx].pushes.len()
                    || shifts[idx].pushes[..pos] != shifts[rep_idx].pushes[..pos]
                    || shifts[idx].pushes[pos + 1..] != shifts[rep_idx].pushes[pos + 1..]
                {
                    continue;
                }

                // This pushed state is buried below an identical pushed suffix
                // and below the current top state. Once buried, it can only be
                // observed by a later reduction querying its goto row, so
                // prefer a predecessor whose goto row is a compatible superset
                // and let the otherwise identical stack paths merge.
                let pred = shifts[idx].pushes[pos];
                let rep = shifts[rep_idx].pushes[pos];
                if goto_row_is_target_compatible_subset(table, pred, rep) {
                    shifts[idx].pushes[pos] = rep;
                    break;
                }
                if goto_row_is_target_compatible_subset(table, rep, pred) {
                    shifts[rep_idx].pushes[pos] = pred;
                    break;
                }
            }
        }
    }
}

fn goto_row_is_target_compatible_subset(table: &GLRTable, subset: u32, superset: u32) -> bool {
    let subset_row = &table.goto[subset as usize];
    let superset_row = &table.goto[superset as usize];
    !subset_row.is_empty()
        && subset_row.iter().all(|(&nt, &target)| {
            superset_row
                .get(&nt)
                .is_some_and(|&superset_target| superset_target == target)
        })
}

fn stack_shift_action(mut shifts: Vec<StackShift>) -> Option<Action> {
    normalize_stack_shifts(&mut shifts);
    if shifts.is_empty() {
        return None;
    }
    if shifts.len() == 1 {
        let shift = &shifts[0];
        if shift.pushes.len() == 1 {
            match shift.pop {
                0 => return Some(Action::Shift(shift.pushes[0], false)),
                1 => return Some(Action::Shift(shift.pushes[0], true)),
                _ => {}
            }
        }
    }
    Some(Action::StackShifts(shifts))
}

