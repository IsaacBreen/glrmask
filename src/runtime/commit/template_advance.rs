use std::collections::HashSet;

use crate::compiler::glr::labels::{
    DEFAULT_LABEL,
    is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::glr::parser::ParserGSS;
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
