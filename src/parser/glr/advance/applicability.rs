pub(crate) fn stack_can_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    for state in stack.peek_values() {
        if !table.advance_row_allows(state, token) {
            continue;
        }

        match table.action(state, token) {
            Some(Action::GuardedStackShifts(shifts)) => {
                let guarded_stack = stack.isolate(Some(state));
                if stack_may_apply_guarded_shifts(&guarded_stack, shifts) {
                    return true;
                }
            }
            Some(_) => return true,
            None => {}
        }
    }

    false
}

fn stack_may_apply_guarded_shifts(stack: &ParserGSS, shifts: &[GuardedStackShift]) -> bool {
    if let Some(virtual_stack) = stack.try_virtual_stack() {
        return shifts
            .iter()
            .any(|shift| virtual_stack_may_apply_guarded_shift(&virtual_stack, shift));
    }

    stack.to_stacks().into_iter().any(|(stack_values, acc)| {
        let single = ParserGSS::from_single_stack(stack_values, acc);
        let Some(virtual_stack) = single.try_virtual_stack() else {
            return false;
        };
        shifts
            .iter()
            .any(|shift| virtual_stack_may_apply_guarded_shift(&virtual_stack, shift))
    })
}
