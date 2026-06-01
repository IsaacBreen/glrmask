use std::collections::HashSet;

use crate::parser::glr::accumulator::TerminalsDisallowed;
use crate::parser::glr::labels::{
    DEFAULT_LABEL,
    is_negative_label,
    negative_to_positive_label,
};
use crate::parser::glr::advance::ParserGSS;
use crate::parser::gss::VirtualStack;
use crate::grammar::flat::TerminalID;
use crate::runtime::CommitTemplateDfas;
use crate::runtime::constraint::Constraint;

pub(super) fn advance_stacks_template_dfa(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: TerminalID,
) -> Option<ParserGSS> {
    let dfa = constraint
        .template_dfas_by_terminal
        .get(terminal as usize)?
        .as_ref()?;
    Some(advance_with_template(dfa, stack.clone()))
}

pub(super) fn advance_stacks_template_dfa_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: TerminalID,
) -> Option<ParserGSS> {
    let dfa = constraint
        .template_dfas_by_terminal
        .get(terminal as usize)?
        .as_ref()?;
    Some(advance_with_template(dfa, stack))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Phase {
    Pop,
    Read,
    Push,
}

fn advance_with_template(template: &CommitTemplateDfas, stack: ParserGSS) -> ParserGSS {
    if let Some(vstack) = stack.try_virtual_stack()
        && let Some(advanced) = advance_virtual_stack_single_path(template, vstack)
    {
        return match advanced {
            SinglePathResult::Unchanged => stack,
            SinglePathResult::Advanced(gss) => gss,
        };
    }

    let mut output = ParserGSS::empty();
    let mut worklist = vec![(Phase::Pop, template.pop.start_state, stack)];
    let mut visited = HashSet::new();

    while let Some((phase, state_id, gss)) = worklist.pop() {
        if gss.is_empty() {
            continue;
        }
        if !visited.insert((phase, state_id, gss.ptr_key())) {
            continue;
        }

        match phase {
            Phase::Pop => {
                let Some(dfa_state) = template.pop.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if is_negative_label(label) {
                        panic!(
                            "commit template pop DFA contains push label {label} at state {state_id}"
                        );
                    }
                    if label != DEFAULT_LABEL && label >= 0 {
                        let state = label as u32;
                        let branch = gss.isolate(Some(state)).popn(1);
                        if !branch.is_empty() {
                            worklist.push((Phase::Pop, target, branch));
                        }
                    }
                }
                if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                    for top in gss.peek_values() {
                        if dfa_state.transitions.contains_key(&(top as i32)) {
                            continue;
                        }
                        let branch = gss.isolate(Some(top)).popn(1);
                        if !branch.is_empty() {
                            worklist.push((Phase::Pop, target, branch));
                        }
                    }
                }

                if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize) {
                    worklist.push((Phase::Read, *read_state, gss.clone()));
                }
                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Read => {
                let Some(dfa_state) = template.read.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if label == DEFAULT_LABEL || is_negative_label(label) {
                        panic!(
                            "commit template read DFA contains non-read label {label} at state {state_id}"
                        );
                    }
                    let branch = gss.isolate(Some(label as u32));
                    if !branch.is_empty() {
                        worklist.push((Phase::Read, target, branch));
                    }
                }

                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    worklist.push((Phase::Push, *push_state, gss));
                }
            }
            Phase::Push => {
                let Some(dfa_state) = template.push.states.get(state_id as usize) else {
                    continue;
                };
                if dfa_state.is_accepting {
                    output = output.merge(&gss);
                }

                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        panic!(
                            "commit template push DFA contains non-push label {label} at state {state_id}"
                        );
                    }
                    worklist.push((
                        Phase::Push,
                        target,
                        gss.push(negative_to_positive_label(label) as u32),
                    ));
                }
            }
        }
    }

    output
}

#[derive(Debug, Clone, Copy)]
enum SingleChoice {
    Pop(u32),
    Read(u32),
    Push(u32, Option<u32>),
}

enum SinglePathResult {
    Unchanged,
    Advanced(ParserGSS),
}

fn advance_virtual_stack_single_path(
    template: &CommitTemplateDfas,
    mut stack: VirtualStack<u32, TerminalsDisallowed>,
) -> Option<SinglePathResult> {
    let mut phase = Phase::Pop;
    let mut state_id = template.pop.start_state;
    let total_states = template
        .pop
        .states
        .len()
        .saturating_add(template.read.states.len())
        .saturating_add(template.push.states.len());
    let max_steps = total_states.saturating_mul(2).saturating_add(8);
    let mut steps = 0usize;
    let mut mutated = false;

    loop {
        let mut choice = None;
        let mut choices = 0usize;
        let accepting;

        match phase {
            Phase::Pop => {
                let dfa_state = template.pop.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                debug_assert!(
                    dfa_state
                        .transitions
                        .keys()
                        .all(|&label| !is_negative_label(label)),
                    "commit template pop DFA contains push label at state {state_id}"
                );

                if let Some(top) = stack.top().copied() {
                    let label = top as i32;
                    if let Some(&target) = dfa_state.transitions.get(&label) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    } else if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                        choice = Some(SingleChoice::Pop(target));
                        choices += 1;
                    }

                    if let Some(Some(read_state)) = template.pop_to_read.get(state_id as usize)
                        && template
                            .read
                            .states
                            .get(*read_state as usize)
                            .is_some_and(|state| state.transitions.contains_key(&label))
                    {
                        choice = Some(SingleChoice::Read(*read_state));
                        choices += 1;
                    }
                }

                if let Some(Some(push_state)) = template.pop_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Read => {
                let dfa_state = template.read.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                debug_assert!(
                    dfa_state
                        .transitions
                        .keys()
                        .all(|&label| label != DEFAULT_LABEL && !is_negative_label(label)),
                    "commit template read DFA contains non-read label at state {state_id}"
                );

                if let Some(top) = stack.top().copied() {
                    let label = top as i32;
                    if let Some(&target) = dfa_state.transitions.get(&label) {
                        choice = Some(SingleChoice::Read(target));
                        choices += 1;
                    }
                }

                if let Some(Some(push_state)) = template.read_to_push.get(state_id as usize) {
                    choice = Some(SingleChoice::Push(*push_state, None));
                    choices += 1;
                }
            }
            Phase::Push => {
                let dfa_state = template.push.states.get(state_id as usize)?;
                accepting = dfa_state.is_accepting;

                for (&label, &target) in &dfa_state.transitions {
                    if !is_negative_label(label) {
                        panic!(
                            "commit template push DFA contains non-push label {label} at state {state_id}"
                        );
                    }
                    choice = Some(SingleChoice::Push(
                        target,
                        Some(negative_to_positive_label(label) as u32),
                    ));
                    choices += 1;
                }
            }
        }

        if choices == 0 {
            if accepting {
                return Some(if mutated {
                    SinglePathResult::Advanced(stack.into_gss())
                } else {
                    SinglePathResult::Unchanged
                });
            }
            return Some(SinglePathResult::Advanced(ParserGSS::empty()));
        }
        if accepting || choices > 1 {
            return None;
        }

        match choice.expect("single applicable split template transition") {
            SingleChoice::Pop(target) => {
                if stack.pop(1) != 0 {
                    return Some(SinglePathResult::Advanced(ParserGSS::empty()));
                }
                mutated = true;
                phase = Phase::Pop;
                state_id = target;
            }
            SingleChoice::Read(target) => {
                phase = Phase::Read;
                state_id = target;
            }
            SingleChoice::Push(target, pushed) => {
                if let Some(pushed) = pushed {
                    stack.push(pushed);
                    mutated = true;
                }
                phase = Phase::Push;
                state_id = target;
            }
        }

        steps += 1;
        if steps > max_steps {
            return None;
        }
    }
}
