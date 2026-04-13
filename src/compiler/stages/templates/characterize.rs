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
    action: &Action,
    escapes: &mut BTreeSet<InitialEscape>,
    reduces: &mut BTreeSet<InitialReduce>,
) {
    match action {
        Action::Shift(shift_state, replace) => {
            let pushes = if *replace { vec![*shift_state] } else { vec![state, *shift_state] };
            escapes.insert((state, pushes));
        }
        Action::Reduce(lhs, len) => {
            let rule_len = *len as usize;
            if rule_len > 0 {
                reduces.insert((state, rule_len - 1, *lhs));
            }
        }
        Action::Split {
            shift,
            reduces: split_reduces,
            ..
        } => {
            if let Some((shift_state, replace)) = shift {
                let pushes = if *replace { vec![*shift_state] } else { vec![state, *shift_state] };
                escapes.insert((state, pushes));
            }
            for &(lhs, len) in split_reduces {
                let rule_len = len as usize;
                if rule_len > 0 {
                    reduces.insert((state, rule_len - 1, lhs));
                }
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
        Action::Reduce(lhs, len) => {
            handle_reduce(
                table,
                stack_nt,
                revealed_state,
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
            record_initial_action(table, state, action, escapes, reduces);
        }
    }

    per_terminal
}

fn characterize_terminal_with_initial(
    table: &GLRTable,
    _grammar: &AnalyzedGrammar,
    terminal: TerminalID,
    escapes: BTreeSet<InitialEscape>,
    reduces: BTreeSet<InitialReduce>,
) -> TerminalCharacterization {
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
    for &(_, _, nt) in &reduces {
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
        reduces: reduces.into_iter().collect(),
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

    while let Some((goto_state, current_replace)) = worklist.pop_front() {
        let Some(action) = table.action(goto_state, terminal) else {
            continue;
        };
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
        0 => unreachable!(),
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
            nt_rereduces.insert((stack_nt, revealed_state, effective_len - 2, reduce_nt));
        }
    }
}
