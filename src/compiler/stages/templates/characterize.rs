//! Terminal characterization for template construction.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{Action, GLRTable};
use crate::grammar::flat::{NonterminalID, TerminalID};

type InitialShift = (u32, u32);

type InitialReduce = (u32, usize, NonterminalID);

type NtEscape = (NonterminalID, u32, u32, u32);

type NtRereduce = (NonterminalID, u32, usize, NonterminalID);

type NtAdjacency = BTreeMap<NonterminalID, BTreeSet<NonterminalID>>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalCharacterization {
    pub shifts: Vec<InitialShift>,
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

fn reduce_rule_info(table: &GLRTable, rule_id: u32) -> (usize, NonterminalID) {
    let rule = &table.rules[rule_id as usize];
    (rule.rhs.len(), rule.lhs)
}

fn record_initial_action(
    table: &GLRTable,
    state: u32,
    action: &Action,
    shifts: &mut BTreeSet<InitialShift>,
    reduces: &mut BTreeSet<InitialReduce>,
) {
    match action {
        Action::Shift(shift_state) => {
            shifts.insert((state, *shift_state));
        }
        Action::Reduce(rule_id) => {
            let (rule_len, lhs) = reduce_rule_info(table, *rule_id);
            if rule_len > 0 {
                reduces.insert((state, rule_len - 1, lhs));
            }
        }
        Action::Split {
            shift,
            reduces: split_reduces,
            ..
        } => {
            if let Some(shift_state) = shift {
                shifts.insert((state, *shift_state));
            }
            for &rule_id in split_reduces {
                let (rule_len, lhs) = reduce_rule_info(table, rule_id);
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
    current_state: u32,
    action: &Action,
    visited: &mut BTreeSet<u32>,
    worklist: &mut VecDeque<u32>,
    nt_escapes: &mut BTreeSet<NtEscape>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
) {
    match action {
        Action::Shift(shift_state) => {
            nt_escapes.insert((stack_nt, revealed_state, current_state, *shift_state));
        }
        Action::Reduce(rule_id) => {
            let (rule_len, lhs) = reduce_rule_info(table, *rule_id);
            handle_reduce(
                table,
                stack_nt,
                revealed_state,
                rule_len,
                lhs,
                visited,
                worklist,
                nt_rereduces,
            );
        }
        Action::Split {
            shift,
            reduces: split_reduces,
            ..
        } => {
            if let Some(shift_state) = shift {
                nt_escapes.insert((stack_nt, revealed_state, current_state, *shift_state));
            }
            for &rule_id in split_reduces {
                let (rule_len, lhs) = reduce_rule_info(table, rule_id);
                handle_reduce(
                    table,
                    stack_nt,
                    revealed_state,
                    rule_len,
                    lhs,
                    visited,
                    worklist,
                    nt_rereduces,
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
        .map(|(terminal, (shifts, reduces))| {
            let terminal = terminal as TerminalID;
            let t0 = std::time::Instant::now();
            let characterization =
                characterize_terminal_with_initial(table, grammar, terminal, shifts, reduces);
            if debug {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let total_entries = characterization.shifts.len()
                    + characterization.reduces.len()
                    + characterization.nt_escapes.len()
                    + characterization.nt_rereduces.len();
                if ms > 1.0 || total_entries > 500 {
                    eprintln!(
                        "[glrmask/debug][characterize_terminal] terminal={} shifts={} reduces={} escapes={} rereduces={} nts={} total={} ms={:.1}",
                        terminal,
                        characterization.shifts.len(),
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
) -> Vec<(BTreeSet<InitialShift>, BTreeSet<InitialReduce>)> {
    let mut per_terminal = vec![(BTreeSet::new(), BTreeSet::new()); num_terminals as usize];

    for state in 0..table.num_states {
        let Some(action_row) = table.action.get(state as usize) else {
            continue;
        };
        for (&terminal, action) in action_row {
            let Some((shifts, reduces)) = per_terminal.get_mut(terminal as usize) else {
                continue;
            };
            record_initial_action(table, state, action, shifts, reduces);
        }
    }

    per_terminal
}

fn characterize_terminal_with_initial(
    table: &GLRTable,
    _grammar: &AnalyzedGrammar,
    terminal: TerminalID,
    shifts: BTreeSet<InitialShift>,
    reduces: BTreeSet<InitialReduce>,
) -> TerminalCharacterization {
    let mut nt_escapes = BTreeSet::new();
    let mut nt_rereduces = BTreeSet::new();

    for revealed_state in 0..table.num_states {
        if let Some(gotos) = table.goto.get(revealed_state as usize) {
            for (&nonterminal, &goto_state) in gotos {
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
                    &mut nt_escapes,
                    &mut nt_rereduces,
                );
            }
        }
    }

    let mut referenced_nts = BTreeSet::new();
    for &(_, _, nt) in &reduces {
        referenced_nts.insert(nt);
    }
    for &(src_nt, _, _, _) in &nt_escapes {
        referenced_nts.insert(src_nt);
    }
    for &(src_nt, _, _, target_nt) in &nt_rereduces {
        referenced_nts.insert(src_nt);
        referenced_nts.insert(target_nt);
    }

    let characterization = TerminalCharacterization {
        shifts: shifts.into_iter().collect(),
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

fn record_goto_action_fast(
    table: &GLRTable,
    stack_nt: NonterminalID,
    revealed_state: u32,
    current_state: u32,
    action: &Action,
    visited: &mut [bool],
    visited_stack: &mut Vec<u32>,
    worklist: &mut VecDeque<u32>,
    nt_escapes: &mut BTreeSet<NtEscape>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
) {
    match action {
        Action::Shift(shift_state) => {
            nt_escapes.insert((stack_nt, revealed_state, current_state, *shift_state));
        }
        Action::Reduce(rule_id) => {
            let (rule_len, lhs) = reduce_rule_info(table, *rule_id);
            handle_reduce_fast(
                table,
                stack_nt,
                revealed_state,
                rule_len,
                lhs,
                visited,
                visited_stack,
                worklist,
                nt_rereduces,
            );
        }
        Action::Split {
            shift,
            reduces: split_reduces,
            ..
        } => {
            if let Some(shift_state) = shift {
                nt_escapes.insert((stack_nt, revealed_state, current_state, *shift_state));
            }
            for &rule_id in split_reduces {
                let (rule_len, lhs) = reduce_rule_info(table, rule_id);
                handle_reduce_fast(
                    table,
                    stack_nt,
                    revealed_state,
                    rule_len,
                    lhs,
                    visited,
                    visited_stack,
                    worklist,
                    nt_rereduces,
                );
            }
        }
        Action::Accept => {}
    }
}

fn handle_reduce_fast(
    table: &GLRTable,
    stack_nt: NonterminalID,
    revealed_state: u32,
    len: usize,
    reduce_nt: NonterminalID,
    visited: &mut [bool],
    visited_stack: &mut Vec<u32>,
    worklist: &mut VecDeque<u32>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
) {
    if len == 1 {
        if let Some(next_goto_state) = table.goto_target(revealed_state, reduce_nt) {
            if (next_goto_state as usize) < visited.len() && !visited[next_goto_state as usize] {
                visited[next_goto_state as usize] = true;
                visited_stack.push(next_goto_state);
                worklist.push_back(next_goto_state);
            }
        }
    } else if len > 1 {
        nt_rereduces.insert((stack_nt, revealed_state, len - 2, reduce_nt));
    }
}

fn explore_from_goto(
    table: &GLRTable,
    terminal: TerminalID,
    stack_nt: NonterminalID,
    revealed_state: u32,
    start_state: u32,
    nt_escapes: &mut BTreeSet<NtEscape>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
) {
    let mut worklist = VecDeque::new();
    let mut visited = BTreeSet::new();

    visited.insert(start_state);
    worklist.push_back(start_state);

    while let Some(current_state) = worklist.pop_front() {
        let Some(action) = table.action(current_state, terminal) else {
            continue;
        };
        record_goto_action(
            table,
            stack_nt,
            revealed_state,
            current_state,
            action,
            &mut visited,
            &mut worklist,
            nt_escapes,
            nt_rereduces,
        );
    }
}

fn handle_reduce(
    table: &GLRTable,
    stack_nt: NonterminalID,
    revealed_state: u32,
    len: usize,
    reduce_nt: NonterminalID,
    visited: &mut BTreeSet<u32>,
    worklist: &mut VecDeque<u32>,
    nt_rereduces: &mut BTreeSet<NtRereduce>,
) {
    if len == 1 {
        if let Some(next_goto_state) = table.goto_target(revealed_state, reduce_nt) {
            if visited.insert(next_goto_state) {
                worklist.push_back(next_goto_state);
            }
        }
    } else if len > 1 {
        nt_rereduces.insert((stack_nt, revealed_state, len - 2, reduce_nt));
    }
}
