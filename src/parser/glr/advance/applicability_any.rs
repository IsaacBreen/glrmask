pub(crate) fn stack_can_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    let top_states = stack.peek_values();
    let mut guarded_terminals = SmallVec::<[TerminalID; 8]>::new();

    for state in top_states {
        if !table.advance_row_intersects(state, terminals) {
            continue;
        }

        if let Some(row) = table.action.get(state as usize)
            && row.len() < terminals.len()
        {
            for (terminal, action) in row {
                let bit = if terminal == EOF {
                    table.num_terminals as usize
                } else {
                    terminal as usize
                };
                if bit > table.num_terminals as usize {
                    continue;
                };
                if !terminals.contains(bit) || !table.advance_row_allows(state, terminal) {
                    continue;
                }

                match action {
                    Action::GuardedStackShifts(_) => {
                        if !guarded_terminals.contains(&terminal) {
                            guarded_terminals.push(terminal);
                        }
                    }
                    _ => return true,
                }
            }
            continue;
        }

        for bit in 0..terminals.len() {
            if !terminals.contains(bit) {
                continue;
            }

            let terminal = if bit == table.num_terminals as usize {
                EOF
            } else {
                bit as TerminalID
            };

            if !table.advance_row_allows(state, terminal) {
                continue;
            }

            match table.action(state, terminal) {
                Some(Action::GuardedStackShifts(_)) => {
                    if !guarded_terminals.contains(&terminal) {
                        guarded_terminals.push(terminal);
                    }
                }
                Some(_) => return true,
                None => {}
            }
        }
    }

    guarded_terminals
        .into_iter()
        .any(|terminal| stack_can_advance_on(table, stack, terminal))
}

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    let has_eof_action = stack
        .peek_values()
        .iter()
        .any(|&state| table.action(state, EOF).is_some());

    has_eof_action
}
