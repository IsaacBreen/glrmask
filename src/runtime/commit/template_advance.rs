use std::collections::HashSet;

use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::labels::{
    DEFAULT_LABEL,
    is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::glr::parser::ParserGSS;
use crate::ds::leveled_gss::VirtualStack;
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

fn advance_with_dfa(dfa: &DFA, stack: ParserGSS) -> ParserGSS {
    if let Some(vstack) = stack.try_virtual_stack()
        && let Some(advanced) = advance_virtual_stack_single_path(dfa, vstack)
    {
        return advanced;
    }

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
            if is_negative_label(label) {
                worklist.push((target, gss.push(negative_to_positive_label(label) as u32)));
            } else if label != DEFAULT_LABEL && label >= 0 {
                let state = label as u32;
                let branch = gss.isolate(Some(state)).popn(1);
                if !branch.is_empty() {
                    worklist.push((target, branch));
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
                    worklist.push((target, branch));
                }
            }
        }
    }

    output
}

fn advance_virtual_stack_single_path(
    dfa: &DFA,
    mut stack: VirtualStack<u32, TerminalsDisallowed>,
) -> Option<ParserGSS> {
    let mut dfa_state_id = dfa.start_state;
    let max_steps = dfa.states.len().saturating_mul(2).saturating_add(8);
    let mut steps = 0usize;

    loop {
        let dfa_state = dfa.states.get(dfa_state_id as usize)?;
        let mut chosen: Option<(i32, u32)> = None;
        let mut applicable = 0usize;

        if let Some(top) = stack.top().copied() {
            let label = top as i32;
            if let Some(&target) = dfa_state.transitions.get(&label) {
                chosen = Some((label, target));
                applicable += 1;
            } else if let Some(&target) = dfa_state.transitions.get(&DEFAULT_LABEL) {
                chosen = Some((DEFAULT_LABEL, target));
                applicable += 1;
            }
        }

        for (&label, &target) in dfa_state.transitions.range(..0) {
            if label == DEFAULT_LABEL {
                continue;
            }
            if is_negative_label(label) {
                chosen = Some((label, target));
                applicable += 1;
            }
        }

        if applicable == 0 {
            return Some(if dfa_state.is_accepting {
                stack.into_gss()
            } else {
                ParserGSS::empty()
            });
        }

        if dfa_state.is_accepting || applicable > 1 {
            return None;
        }

        let (label, target) = chosen.expect("single applicable transition");
        if label == DEFAULT_LABEL || label >= 0 {
            if stack.pop(1) != 0 {
                return Some(ParserGSS::empty());
            }
        } else if is_negative_label(label) {
            stack.push(negative_to_positive_label(label) as u32);
        } else {
            return None;
        }

        dfa_state_id = target;
        steps += 1;
        if steps > max_steps {
            return None;
        }
    }
}
