//! Terminal characterization for template construction.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{Action, GLRTable};
use crate::grammar::flat::{NonterminalID, TerminalID};

type InitialEscape = (u32, Vec<u32>);

type InitialReduce = (u32, usize, NonterminalID);

type NtEscape = (NonterminalID, u32, Vec<u32>);

type NtRereduce = (NonterminalID, u32, usize, NonterminalID);

type NtAdjacency = BTreeMap<NonterminalID, BTreeSet<NonterminalID>>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalCharacterization {
    pub escapes: Vec<InitialEscape>,
    pub reduces: Vec<InitialReduce>,
    pub nt_escapes: Vec<NtEscape>,
    pub nt_rereduces: Vec<NtRereduce>,
    pub all_nts: BTreeSet<NonterminalID>,
}

impl TerminalCharacterization {
    pub fn find_cycle(&self) -> Option<Vec<NonterminalID>> {
        find_nonterminal_cycle(&build_rereduce_adjacency(&self.nt_rereduces))
    }
}

fn build_rereduce_adjacency(nt_rereduces: &[NtRereduce]) -> NtAdjacency {
    let mut adjacency = NtAdjacency::new();
    for (source_nonterminal, _revealed_state, _pop_count, target_nonterminal) in nt_rereduces {
        adjacency
            .entry(*source_nonterminal)
            .or_default()
            .insert(*target_nonterminal);
    }
    adjacency
}

fn dfs_nonterminal_cycle(
    nonterminal: NonterminalID,
    adjacency: &NtAdjacency,
    colors: &mut BTreeMap<NonterminalID, u8>,
    path: &mut Vec<NonterminalID>,
) -> Option<Vec<NonterminalID>> {
    colors.insert(nonterminal, 1);
    path.push(nonterminal);

    if let Some(neighbors) = adjacency.get(&nonterminal) {
        for &neighbor in neighbors {
            match colors.get(&neighbor).copied().unwrap_or(0) {
                1 => {
                    let cycle_start = path.iter().position(|nt| *nt == neighbor).unwrap_or(0);
                    let mut cycle = path[cycle_start..].to_vec();
                    cycle.push(neighbor);
                    return Some(cycle);
                }
                0 => {
                    if let Some(cycle) = dfs_nonterminal_cycle(neighbor, adjacency, colors, path) {
                        return Some(cycle);
                    }
                }
                _ => {}
            }
        }
    }

    path.pop();
    colors.insert(nonterminal, 2);
    None
}

fn find_nonterminal_cycle(adjacency: &NtAdjacency) -> Option<Vec<NonterminalID>> {
    let mut colors = BTreeMap::new();
    let mut path = Vec::new();

    for &nonterminal in adjacency.keys() {
        if colors.get(&nonterminal).copied().unwrap_or(0) == 0 {
            if let Some(cycle) = dfs_nonterminal_cycle(nonterminal, adjacency, &mut colors, &mut path) {
                return Some(cycle);
            }
        }
    }

    None
}

fn record_initial_action(
    _table: &GLRTable,
    state: u32,
    _terminal: TerminalID,
    action: &Action,
    is_forwarded: bool,
    escapes: &mut BTreeSet<InitialEscape>,
    reduces: &mut BTreeSet<InitialReduce>,
) {
    match action {
        Action::Shift(shift_state, replace) => {
            let effective_replace = *replace;
            let pushes = if effective_replace { vec![*shift_state] } else { vec![state, *shift_state] };
            escapes.insert((state, pushes));
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                if shift.pop <= 1 {
                    let mut pushes = if shift.pop == 0 { vec![state] } else { Vec::new() };
                    pushes.extend_from_slice(&shift.pushes);
                    escapes.insert((state, pushes));
                }
            }
        }
        Action::GuardedStackShifts(_) => {}
        Action::Reduce(lhs, len) => {
            let rule_len = if is_forwarded { (*len as usize) + 1 } else { *len as usize };
            reduces.insert((state, rule_len, *lhs));
        }
        Action::Split {
            shift,
            reduces: split_reduces,
            ..
        } => {
            if let Some((shift_state, replace)) = shift {
                let effective_replace = *replace;
                let pushes = if effective_replace { vec![*shift_state] } else { vec![state, *shift_state] };
                escapes.insert((state, pushes));
            }
            for &(lhs, len) in split_reduces {
                let rule_len = if is_forwarded { (len as usize) + 1 } else { len as usize };
                reduces.insert((state, rule_len, lhs));
            }
        }
        Action::Accept => {}
    }
}

fn record_goto_action(
    table: &GLRTable,
    stack_nt: NonterminalID,
    revealed_state: u32,
    goto_state: u32,
    action: &Action,
    goto_replace: bool,
    visited: &mut BTreeSet<(u32, bool)>,
    worklist: &mut VecDeque<(u32, bool)>,
    nt_escapes: &mut BTreeSet<NtEscape>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
    inheritances: &mut BTreeMap<u32, BTreeSet<u32>>,
) {
    match action {
        Action::Shift(shift_state, shift_replace) => {
            let mut pushes = Vec::new();
            if !goto_replace { pushes.push(revealed_state); }
            if !*shift_replace { pushes.push(goto_state); }
            pushes.push(*shift_state);
            nt_escapes.insert((stack_nt, revealed_state, pushes));
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                let mut pushes = Vec::new();
                if !goto_replace {
                    pushes.push(revealed_state);
                }
                pushes.push(goto_state);
                if shift.pop as usize <= pushes.len() {
                    pushes.truncate(pushes.len() - shift.pop as usize);
                    pushes.extend_from_slice(&shift.pushes);
                    nt_escapes.insert((stack_nt, revealed_state, pushes));
                }
            }
        }
        Action::GuardedStackShifts(_) => {}
        Action::Reduce(lhs, len) => {
            handle_reduce(
                table,
                stack_nt,
                revealed_state,
                goto_state,
                *len as usize,
                *lhs,
                goto_replace,
                visited,
                worklist,
                nt_rereduces,
                inheritances,
            );
        }
        Action::Split {
            shift,
            reduces: split_reduces,
            ..
        } => {
            if let Some((shift_state, shift_replace)) = shift {
                let mut pushes = Vec::new();
                if !goto_replace { pushes.push(revealed_state); }
                if !*shift_replace { pushes.push(goto_state); }
                pushes.push(*shift_state);
                nt_escapes.insert((stack_nt, revealed_state, pushes));
            }
            for &(lhs, len) in split_reduces {
                handle_reduce(
                    table,
                    stack_nt,
                    revealed_state,
                    goto_state,
                    len as usize,
                    lhs,
                    goto_replace,
                    visited,
                    worklist,
                    nt_rereduces,
                    inheritances,
                );
            }
        }
        Action::Accept => {}
    }
}

pub(crate) fn characterize_terminals(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
) -> BTreeMap<TerminalID, TerminalCharacterization> {
    let debug = std::env::var("GLRMASK_DEBUG_PROFILE").is_ok();
    let initial_actions = collect_initial_actions_by_terminal(table, grammar.num_terminals);

    if debug {
        let total_gotos: usize = table.goto.iter().map(|g| g.len()).sum();
        eprintln!(
            "[glrmask/debug][characterize] glr_states={} total_gotos={} num_terminals={}",
            table.num_states, total_gotos, grammar.num_terminals
        );
    }

    initial_actions
        .into_iter()
        .enumerate()
        .map(|(terminal, (escapes, reduces))| {
            let terminal = terminal as TerminalID;
            let t0 = std::time::Instant::now();
            let characterization =
                characterize_terminal_with_initial(table, grammar, terminal, escapes, reduces);
            if debug {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let total_entries = characterization.escapes.len()
                    + characterization.reduces.len()
                    + characterization.nt_escapes.len()
                    + characterization.nt_rereduces.len();
                if ms > 1.0 || total_entries > 500 {
                    eprintln!(
                        "[glrmask/debug][characterize_terminal] terminal={} escapes={} reduces={} nt_escapes={} rereduces={} nts={} total={} ms={:.1}",
                        terminal,
                        characterization.escapes.len(),
                        characterization.reduces.len(),
                        characterization.nt_escapes.len(),
                        characterization.nt_rereduces.len(),
                        characterization.all_nts.len(),
                        total_entries,
                        ms,
                    );
                }
            }
            (terminal, characterization)
        })
        .collect()
}

fn collect_initial_actions_by_terminal(
    table: &GLRTable,
    num_terminals: u32,
) -> Vec<(BTreeSet<InitialEscape>, BTreeSet<InitialReduce>)> {
    let mut per_terminal = vec![(BTreeSet::new(), BTreeSet::new()); num_terminals as usize];

    for state in 0..table.num_states {
        let Some(action_row) = table.action.get(state as usize) else {
            continue;
        };
        for (&terminal, action) in action_row {
            let Some((escapes, reduces)) = per_terminal.get_mut(terminal as usize) else {
                continue;
            };
            let is_forwarded = table.forwarded_shifts.contains(&(state, terminal));
            record_initial_action(table, state, terminal, action, is_forwarded, escapes, reduces);
        }
    }

    per_terminal
}

/// Expand a pop-0 initial reduce by following the goto chain.
///
/// Tracks a `push_stack` of states pushed by non-replace gotos during
/// the chain.  When a shift is reached, emits an initial escape.  When a
/// reduce pops more states than were pushed, emits a normal initial
/// reduce with the excess as its pop count.
fn expand_zero_pop_initial_reduce(
    table: &GLRTable,
    terminal: TerminalID,
    initial_state: u32,
    nonterminal: NonterminalID,
    escapes: &mut BTreeSet<InitialEscape>,
    reduces: &mut BTreeSet<InitialReduce>,
    visited: &mut BTreeSet<(u32, NonterminalID)>,
) {
    if !visited.insert((initial_state, nonterminal)) {
        return; // cycle
    }

    let mut push_stack: Vec<u32> = Vec::new();
    expand_zero_pop_chain(
        table,
        terminal,
        initial_state,
        nonterminal,
        &mut push_stack,
        escapes,
        reduces,
        visited,
    );
}

fn expand_zero_pop_chain(
    table: &GLRTable,
    terminal: TerminalID,
    initial_state: u32,
    nonterminal: NonterminalID,
    push_stack: &mut Vec<u32>,
    escapes: &mut BTreeSet<InitialEscape>,
    reduces: &mut BTreeSet<InitialReduce>,
    visited: &mut BTreeSet<(u32, NonterminalID)>,
) {
    // Follow the goto for `nonterminal` from the current "top" state.
    let current_state = push_stack.last().copied().unwrap_or(initial_state);
    let Some(&(goto_state, goto_replace)) = table
        .goto
        .get(current_state as usize)
        .and_then(|g| g.get(&nonterminal))
    else {
        return;
    };

    // Track the goto's push.
    if !goto_replace {
        push_stack.push(goto_state);
    } else {
        // Replace goto: the goto_state replaces the current top.
        // If push_stack is non-empty, replace the top. Otherwise, the
        // initial_state is conceptually replaced.
        if let Some(top) = push_stack.last_mut() {
            *top = goto_state;
        }
        // If push_stack is empty and goto replaces initial_state, we model
        // this by not pushing (the replace consumed the initial_state slot).
    }

    let Some(action) = table.action(goto_state, terminal) else {
        // Undo the push for clean state if we need to backtrack.
        if !goto_replace && !push_stack.is_empty() {
            push_stack.pop();
        }
        return;
    };

    expand_zero_pop_action(
        table, terminal, initial_state, action, goto_state, goto_replace,
        push_stack, escapes, reduces, visited,
    );

    // Undo the push so that the caller's push_stack is unchanged.
    // (Each branch of a Split gets its own clone, but for non-split we
    //  need to restore.)
    // Actually, since we only call this once per nonterminal and don't
    // return to use push_stack again in the caller, this is fine.
}

fn expand_zero_pop_action(
    table: &GLRTable,
    terminal: TerminalID,
    initial_state: u32,
    action: &Action,
    _goto_state: u32,
    _goto_replace: bool,
    push_stack: &mut Vec<u32>,
    escapes: &mut BTreeSet<InitialEscape>,
    reduces: &mut BTreeSet<InitialReduce>,
    visited: &mut BTreeSet<(u32, NonterminalID)>,
) {
    match action {
        Action::Shift(shift_state, shift_replace) => {
            // Build the escape pushes from the accumulated push_stack.
            let mut pushes = Vec::new();
            pushes.push(initial_state); // re-push initial (since positive(initial) will pop it)
            pushes.extend_from_slice(push_stack);
            if !*shift_replace {
                // The shift pushes its target; if replace, it replaces the
                // top which is already in push_stack (or goto_state).
                pushes.push(*shift_state);
            } else {
                // Replace shift: replace the top of pushes with shift_state.
                if let Some(top) = pushes.last_mut() {
                    *top = *shift_state;
                }
            }
            escapes.insert((initial_state, pushes));
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                let mut pushes = Vec::new();
                pushes.push(initial_state);
                pushes.extend_from_slice(push_stack);
                if shift.pop as usize <= pushes.len() {
                    pushes.truncate(pushes.len() - shift.pop as usize);
                    pushes.extend_from_slice(&shift.pushes);
                    escapes.insert((initial_state, pushes));
                }
            }
        }
        Action::GuardedStackShifts(_) => {}
        Action::Reduce(nt2, len2) => {
            handle_zero_pop_reduce(
                table, terminal, initial_state, *nt2, *len2 as usize,
                push_stack, escapes, reduces, visited,
            );
        }
        Action::Split {
            shift,
            reduces: split_reduces,
            ..
        } => {
            if let Some((shift_state, shift_replace)) = shift {
                let mut pushes = Vec::new();
                pushes.push(initial_state);
                pushes.extend_from_slice(push_stack);
                if !*shift_replace {
                    pushes.push(*shift_state);
                } else {
                    if let Some(top) = pushes.last_mut() {
                        *top = *shift_state;
                    }
                }
                escapes.insert((initial_state, pushes));
            }
            for &(nt2, len2) in split_reduces {
                let mut ps = push_stack.clone();
                handle_zero_pop_reduce(
                    table, terminal, initial_state, nt2, len2 as usize,
                    &mut ps, escapes, reduces, visited,
                );
            }
        }
        Action::Accept => {}
    }
}

fn handle_zero_pop_reduce(
    table: &GLRTable,
    terminal: TerminalID,
    initial_state: u32,
    reduce_nt: NonterminalID,
    reduce_len: usize,
    push_stack: &mut Vec<u32>,
    escapes: &mut BTreeSet<InitialEscape>,
    reduces: &mut BTreeSet<InitialReduce>,
    visited: &mut BTreeSet<(u32, NonterminalID)>,
) {
    if reduce_len <= push_stack.len() {
        // The reduce pops within the pushed states — no net NFA pop needed.
        push_stack.truncate(push_stack.len() - reduce_len);
        // Continue by following the goto for reduce_nt from the new "top".
        if !visited.contains(&(push_stack.last().copied().unwrap_or(initial_state), reduce_nt)) {
            expand_zero_pop_chain(
                table, terminal, initial_state, reduce_nt,
                push_stack, escapes, reduces, visited,
            );
        }
    } else {
        // The reduce pops more than we've pushed — the excess becomes the
        // NFA pop count. Record as a normal initial reduce.
        let nfa_pops = reduce_len - push_stack.len();
        reduces.insert((initial_state, nfa_pops, reduce_nt));
    }
}

fn characterize_terminal_with_initial(
    table: &GLRTable,
    _grammar: &AnalyzedGrammar,
    terminal: TerminalID,
    mut escapes: BTreeSet<InitialEscape>,
    reduces: BTreeSet<InitialReduce>,
) -> TerminalCharacterization {
    // Expand pop-0 initial reduces by following the goto chain.
    // Pop-0 reduces become epsilon transitions in the template NFA, which
    // bypass stack checks and make the mask too permissive. By inlining
    // the goto chain, we convert them to equivalent escapes or higher-pop
    // reduces that correctly gate on the parser state.
    let mut normal_reduces: BTreeSet<InitialReduce> = BTreeSet::new();
    for &(initial_state, pop_count, nonterminal) in &reduces {
        if pop_count == 0 {
            let mut visited = BTreeSet::new();
            expand_zero_pop_initial_reduce(
                table,
                terminal,
                initial_state,
                nonterminal,
                &mut escapes,
                &mut normal_reduces,
                &mut visited,
            );
        } else {
            normal_reduces.insert((initial_state, pop_count, nonterminal));
        }
    }

    let mut nt_escapes = BTreeSet::new();
    let mut nt_rereduces = BTreeSet::new();
    let mut inheritances: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();

    for revealed_state in 0..table.num_states {
        if let Some(gotos) = table.goto.get(revealed_state as usize) {
            for (&nonterminal, &(goto_state, goto_replace)) in gotos {
                // Skip traversals where the goto target has no action for this
                // terminal — the BFS would immediately return with no effect.
                if table.action(goto_state, terminal).is_none() {
                    continue;
                }
                explore_from_goto(
                    table,
                    terminal,
                    nonterminal,
                    revealed_state,
                    goto_state,
                    goto_replace,
                    &mut nt_escapes,
                    &mut nt_rereduces,
                    &mut inheritances,
                );
            }
        }
    }

    // Debug: print nt_escapes before inheritance
    if std::env::var("GLRMASK_DEBUG_CHARACTERIZE").is_ok() {
        eprintln!("[debug characterize] terminal={} nt_escapes_before_inherit={:?}", terminal, nt_escapes);
        eprintln!("[debug characterize] terminal={} inheritances={:?}", terminal, inheritances);
    }

    // Distribute inheritances: when revealed_state R inherits from R2 (due to
    // a replace goto R → R2), copy all nt_escapes and nt_rereduces that have
    // R2 as their revealed_state to also use R.  Repeat until no new entries.
    if !inheritances.is_empty() {
        loop {
            let mut new_escapes = BTreeSet::new();
            let mut new_rereduces = BTreeSet::new();

            for (&inheritor, targets) in &inheritances {
                for &target in targets {
                    // Copy nt_escapes: (nt, target, pushes) → (nt, inheritor, pushes)
                    for &(nt, revealed, ref pushes) in &nt_escapes {
                        if revealed == target {
                            let inherited = (nt, inheritor, pushes.clone());
                            if !nt_escapes.contains(&inherited) {
                                new_escapes.insert(inherited);
                            }
                        }
                    }
                    // Copy nt_rereduces: (nt, target, pop, tgt_nt) → (nt, inheritor, pop, tgt_nt)
                    for &(nt, revealed, pop, tgt_nt) in &nt_rereduces {
                        if revealed == target {
                            let inherited = (nt, inheritor, pop, tgt_nt);
                            if !nt_rereduces.contains(&inherited) {
                                new_rereduces.insert(inherited);
                            }
                        }
                    }
                }
            }

            if new_escapes.is_empty() && new_rereduces.is_empty() {
                break;
            }
            nt_escapes.extend(new_escapes);
            nt_rereduces.extend(new_rereduces);
        }
    }

    let mut referenced_nts = BTreeSet::new();
    for &(_, _, nt) in &normal_reduces {
        referenced_nts.insert(nt);
    }
    for &(src_nt, _, _) in &nt_escapes {
        referenced_nts.insert(src_nt);
    }
    for &(src_nt, _, _, target_nt) in &nt_rereduces {
        referenced_nts.insert(src_nt);
        referenced_nts.insert(target_nt);
    }

    let characterization = TerminalCharacterization {
        escapes: escapes.into_iter().collect(),
        reduces: normal_reduces.into_iter().collect(),
        nt_escapes: nt_escapes.into_iter().collect(),
        nt_rereduces: nt_rereduces.into_iter().collect(),
        all_nts: referenced_nts,
    };

    if let Some(cycle) = characterization.find_cycle() {
        panic!(
            "terminal characterization for terminal {} contains a reduction cycle: {:?}",
            terminal,
            cycle
        );
    }

    characterization
}

fn explore_from_goto(
    table: &GLRTable,
    terminal: TerminalID,
    stack_nt: NonterminalID,
    revealed_state: u32,
    start_state: u32,
    goto_replace: bool,
    nt_escapes: &mut BTreeSet<NtEscape>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
    inheritances: &mut BTreeMap<u32, BTreeSet<u32>>,
) {
    let mut worklist = VecDeque::new();
    let mut visited = BTreeSet::new();

    visited.insert((start_state, goto_replace));
    worklist.push_back((start_state, goto_replace));

    let debug_char = std::env::var("GLRMASK_DEBUG_CHARACTERIZE").is_ok();
    while let Some((goto_state, current_replace)) = worklist.pop_front() {
        let Some(action) = table.action(goto_state, terminal) else {
            continue;
        };
        if debug_char && stack_nt == 1 && revealed_state == 0 {
            eprintln!("[debug N1@0] BFS pop goto_state={} replace={} action={:?}", goto_state, current_replace, action);
        }
        record_goto_action(
            table,
            stack_nt,
            revealed_state,
            goto_state,
            action,
            current_replace,
            &mut visited,
            &mut worklist,
            nt_escapes,
            nt_rereduces,
            inheritances,
        );
    }
}

fn handle_reduce(
    table: &GLRTable,
    stack_nt: NonterminalID,
    revealed_state: u32,
    goto_state: u32,
    len: usize,
    reduce_nt: NonterminalID,
    goto_replace: bool,
    visited: &mut BTreeSet<(u32, bool)>,
    worklist: &mut VecDeque<(u32, bool)>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
    inheritances: &mut BTreeMap<u32, BTreeSet<u32>>,
) {
    // If the goto was a replace, the goto state was never pushed, so the
    // reduce effectively pops one fewer item from the perspective of the
    // template NFA.  Equivalently, add 1 to len.
    let effective_len = if goto_replace { len + 1 } else { len };
    match effective_len {
        0 => {
            // Zero-length reduce at a non-replace goto target.
            // The goto_state is still on the stack; follow the goto from it.
            if let Some((next_goto_state, next_replace)) = table.goto_target(goto_state, reduce_nt) {
                if visited.insert((next_goto_state, next_replace)) {
                    worklist.push_back((next_goto_state, next_replace));
                }
            }
        }
        1 => {
            if let Some((next_goto_state, next_replace)) = table.goto_target(revealed_state, reduce_nt) {
                if next_replace {
                    // The goto from revealed_state is replace — revealed_state
                    // inherits all escapes/rereduces of next_goto_state.
                    inheritances
                        .entry(revealed_state)
                        .or_default()
                        .insert(next_goto_state);
                }
                if visited.insert((next_goto_state, next_replace)) {
                    worklist.push_back((next_goto_state, next_replace));
                }
            }
        }
        2.. => {
            nt_rereduces.insert((stack_nt, revealed_state, effective_len - 1, reduce_nt));
        }
    }
}
