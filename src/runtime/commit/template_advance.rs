use std::collections::HashSet;

use crate::compiler::glr::labels::{
    DEFAULT_LABEL,
    is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::glr::parser::ParserGSS;
use crate::grammar::flat::TerminalID;
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
    Some(advance_with_dfa(dfa, stack.clone()))
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
    Some(advance_with_dfa(dfa, stack))
}

fn advance_with_dfa(
    dfa: &crate::automata::unweighted_u32::dfa::DFA,
    stack: ParserGSS,
) -> ParserGSS {
    let mut output = ParserGSS::empty();
    let mut worklist = vec![(dfa.start_state, stack)];
    let mut visited = HashSet::new();

    while let Some((dfa_state_id, gss)) = worklist.pop() {
        if gss.is_empty() {
            continue;
        }
        if !visited.insert((dfa_state_id, gss.ptr_key())) {
            continue;
        }

        let Some(dfa_state) = dfa.states.get(dfa_state_id as usize) else {
            continue;
        };
        if dfa_state.is_accepting {
            output = output.merge(&gss);
        }

        for (&label, &target) in &dfa_state.transitions {
            if label == DEFAULT_LABEL {
                for top in gss.peek_values() {
                    let branch = gss.isolate(Some(top)).popn(1);
                    if !branch.is_empty() {
                        worklist.push((target, branch));
                    }
                }
            } else if is_negative_label(label) {
                worklist.push((target, gss.push(negative_to_positive_label(label) as u32)));
            } else if label >= 0 {
                let state = label as u32;
                let branch = gss.isolate(Some(state)).popn(1);
                if !branch.is_empty() {
                    worklist.push((target, branch));
                }
            }
        }
    }

    output
}
