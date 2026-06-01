//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

pub(super) fn state_has_nonempty_accumulators(state: &BTreeMap<u32, ParserGSS>) -> bool {
    state
        .values()
        .any(|gss| !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()))
}

pub(super) fn end_state_can_advance(constraint: &Constraint, gss: &ParserGSS, end_state: u32) -> bool {
    end_state == constraint.tokenizer.initial_state()
        || stack_can_advance_on_any(
            &constraint.table,
            gss,
            constraint.tokenizer.possible_future_terminals(end_state),
        )
}

pub(super) fn prune_initial_states(
    state: &mut BTreeMap<u32, ParserGSS>,
    accepted_terminals: &FxHashMap<u32, FxHashSet<u32>>,
    remapped_tokenizer_states: &FxHashMap<u32, u32>,
) {
    if accepted_terminals.is_empty()
        && state
            .values()
            .all(|parser_state| parser_state.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()))
    {
        return;
    }

    for parser_state in state.values_mut() {
        *parser_state = parser_state.apply_and_prune_no_promote(
            |terminals_disallowed: &TerminalsDisallowed| {
                for (tokenizer_state, matched_terminals) in accepted_terminals {
                    if let Some(disallowed) = terminals_disallowed.get(tokenizer_state) {
                        if !matched_terminals.is_empty()
                            && matched_terminals
                                .iter()
                                .all(|terminal| disallowed.contains(terminal))
                        {
                            return None;
                        }
                    }
                }

                let mut remapped = BTreeMap::new();
                for (old_state, new_state) in remapped_tokenizer_states {
                    if let Some(disallowed) = terminals_disallowed.get(old_state) {
                        remapped
                            .entry(*new_state)
                            .or_insert_with(BTreeSet::new)
                            .extend(disallowed.iter().copied());
                    }
                }
                Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
            },
        );
    }
}

pub(super) fn prune_single_initial_state_for_exec(
    constraint: &Constraint,
    gss: ParserGSS,
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
) -> ParserGSS {
    let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss);
    let mut accepted_terminals: SmallVec<[u32; 4]> = SmallVec::new();
    for matched in &exec_result.matches {
        if is_ignored_terminal(constraint.ignore_terminal, matched.id) {
            continue;
        }
        if is_actionable_terminal(actionable_terminals.as_ref(), constraint, matched.id) {
            accepted_terminals.push(matched.id);
        }
    }

    if accepted_terminals.is_empty() && exec_result.end_state.is_none() {
        return gss;
    }

    if accepted_terminals.is_empty()
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
    {
        return gss;
    }

    gss.apply_and_prune_no_promote(|terminals_disallowed: &TerminalsDisallowed| {
        if terminals_disallowed.is_empty() {
            return Some(TerminalsDisallowed::new());
        }
        if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
            if !accepted_terminals.is_empty()
                && accepted_terminals
                    .iter()
                    .all(|terminal| disallowed.contains(terminal))
            {
                return None;
            }
        }

        let mut remapped = BTreeMap::new();
        if let Some(end_state) = exec_result.end_state {
            if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
                remapped
                    .entry(end_state)
                    .or_insert_with(BTreeSet::new)
                    .extend(disallowed.iter().copied());
            }
        }
        Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
    })
}

pub(super) fn prune_single_initial_state_for_terminal(
    gss: ParserGSS,
    tokenizer_state: u32,
    terminal: u32,
    end_state: Option<u32>,
) -> ParserGSS {
    if end_state.is_none()
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| {
            td.get(&tokenizer_state)
                .is_none_or(|disallowed| !disallowed.contains(&terminal))
        })
    {
        return gss.apply(|_: &TerminalsDisallowed| TerminalsDisallowed::new());
    }

    gss.apply_and_prune_no_promote(|terminals_disallowed: &TerminalsDisallowed| {
        if terminals_disallowed.is_empty() {
            return Some(TerminalsDisallowed::new());
        }
        if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
            if disallowed.contains(&terminal) {
                return None;
            }
        }

        let mut remapped = BTreeMap::new();
        if let Some(end_state) = end_state {
            if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
                remapped
                    .entry(end_state)
                    .or_insert_with(BTreeSet::new)
                    .extend(disallowed.iter().copied());
            }
        }
        Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
    })
}

pub(super) fn apply_future_terminal_disallow(
    constraint: &Constraint,
    exec_result: &TokenizerExecResult,
    terminal: u32,
    gss: ParserGSS,
) -> ParserGSS {
    if gss.is_empty() {
        return gss;
    }

    let Some(end_state) = exec_result.end_state else {
        return gss;
    };
    if !constraint
        .tokenizer
        .dfa
        .possible_future_group_ids(end_state)
        .contains(terminal as usize)
    {
        return gss;
    }

    gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
        terminals_disallowed.with_insert(end_state, terminal)
    })
}

