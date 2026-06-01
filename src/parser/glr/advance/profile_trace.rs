fn trace_action_kind(action: Option<&Action>) -> &'static str {
    match action {
        Some(Action::Shift(..)) => "shift",
        Some(Action::StackShifts(..)) => "stack-shifts",
        Some(Action::GuardedStackShifts(..)) => "guarded-stack-shifts",
        Some(Action::Reduce(..)) => "reduce",
        Some(Action::Split { accept: true, .. }) => "split-accept",
        Some(Action::Split { .. }) => "split",
        Some(Action::Accept) => "accept",
        None => "none",
    }
}

fn trace_reduce_summary(
    table: &GLRTable,
    gss: &ParserGSS,
    lhs_nt: u32,
    pop_len: usize,
) -> AdvanceTraceReduce {
    let mut goto_sources = Vec::new();
    let mut goto_targets = Vec::new();
    for (goto_from, _) in reduce_sources_from_isolated(gss, pop_len) {
        goto_sources.push(goto_from);
        if let Some((target_state, replace)) = table.goto_target(goto_from, lhs_nt) {
            goto_targets.push(AdvanceTraceGoto {
                source_state: goto_from,
                target_state,
                replace,
            });
        }
    }
    goto_sources.sort_unstable();
    goto_sources.dedup();
    goto_targets.sort_by_key(|entry| (entry.source_state, entry.target_state, entry.replace));
    goto_targets.dedup_by(|left, right| {
        left.source_state == right.source_state
            && left.target_state == right.target_state
            && left.replace == right.replace
    });
    AdvanceTraceReduce {
        lhs_nt,
        lhs_name: table.nonterminal_display_name(lhs_nt).map(str::to_owned),
        pop_len: pop_len as u32,
        goto_sources,
        goto_targets,
    }
}

fn trace_action_summary(
    table: &GLRTable,
    source_state: u32,
    gss: &ParserGSS,
    action: Option<&Action>,
) -> AdvanceTraceStep {
    match action {
        Some(Action::Shift(target, replace)) => AdvanceTraceStep {
            source_state,
            action_kind: trace_action_kind(action).to_string(),
            shift_target: Some(*target),
            shift_replace: Some(*replace),
            reduces: Vec::new(),
        },
        Some(Action::StackShifts(..)) | Some(Action::GuardedStackShifts(..)) | Some(Action::Accept) | None => {
            AdvanceTraceStep {
                source_state,
                action_kind: trace_action_kind(action).to_string(),
                shift_target: None,
                shift_replace: None,
                reduces: Vec::new(),
            }
        }
        Some(Action::Reduce(lhs_nt, pop_len)) => AdvanceTraceStep {
            source_state,
            action_kind: trace_action_kind(action).to_string(),
            shift_target: None,
            shift_replace: None,
            reduces: vec![trace_reduce_summary(table, gss, *lhs_nt, *pop_len as usize)],
        },
        Some(Action::Split { shift, reduces, accept }) => AdvanceTraceStep {
            source_state,
            action_kind: if *accept { "split-accept" } else { "split" }.to_string(),
            shift_target: shift.map(|(target, _)| target),
            shift_replace: shift.map(|(_, replace)| replace),
            reduces: reduces
                .iter()
                .map(|&(lhs_nt, pop_len)| trace_reduce_summary(table, gss, lhs_nt, pop_len as usize))
                .collect(),
        },
    }
}

enum AdvancedBranch {
    Stack(VirtualStack<u32, TerminalsDisallowed>),
    Gss(ParserGSS),
}

impl AdvancedBranch {
    fn into_gss(self) -> ParserGSS {
        match self {
            AdvancedBranch::Stack(stack) => stack.into_gss(),
            AdvancedBranch::Gss(gss) => gss,
        }
    }
}
