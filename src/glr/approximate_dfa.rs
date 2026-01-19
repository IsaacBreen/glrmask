use crate::datastructures::bitset::Bitset;
use crate::glr::parser::GLRParser;
use crate::glr::table::{Stage7ShiftsAndReducesLookaheadValue, StateID, Table, TerminalID, NonTerminalID};
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone)]
pub struct ApproximateParserNFA {
    pub num_states: usize,
    pub start_state: StateID,
    pub transitions: Vec<BTreeMap<TerminalID, Bitset>>,
}

#[derive(Debug, Clone)]
pub struct ApproximateParserDFA {
    pub start_state: usize,
    pub transitions: Vec<BTreeMap<TerminalID, usize>>,
    pub dfa_state_sets: Vec<Bitset>,
}

impl ApproximateParserDFA {
    pub fn step(&self, state: usize, terminal: TerminalID) -> Option<usize> {
        self.transitions
            .get(state)
            .and_then(|map| map.get(&terminal).copied())
    }
}

pub fn build_approximate_parser_dfa(parser: &GLRParser) -> ApproximateParserDFA {
    let num_states = table_state_count(&parser.table);
    let underneath_map = compute_underneath_map(&parser.table, num_states);
    let nfa = build_nfa(parser, num_states, &underneath_map);
    determinize_nfa(&nfa)
}

fn table_state_count(table: &Table) -> usize {
    table.keys().map(|s| s.0).max().unwrap_or(0) + 1
}

fn compute_underneath_map(table: &Table, num_states: usize) -> Vec<Bitset> {
    let mut underneath = vec![Bitset::new(num_states); num_states];
    for (&state_id, row) in table.iter() {
        for action in row.get_shifts_and_reduces_map().values() {
            if let Stage7ShiftsAndReducesLookaheadValue::Shift(next_state) = action {
                underneath[next_state.0].insert(state_id.0);
            } else if let Stage7ShiftsAndReducesLookaheadValue::Split { shift, .. } = action {
                if let Some(next_state) = shift {
                    underneath[next_state.0].insert(state_id.0);
                }
            }
        }

        for goto in row.get_gotos().values() {
            if let Some(next_state) = goto.state_id {
                underneath[next_state.0].insert(state_id.0);
            }
        }
    }
    underneath
}

fn build_nfa(parser: &GLRParser, num_states: usize, underneath_map: &[Bitset]) -> ApproximateParserNFA {
    let num_terminals = parser.terminal_map.len();
    let mut transitions: Vec<BTreeMap<TerminalID, Bitset>> = vec![BTreeMap::new(); num_states];

    let mut goto_map: Vec<BTreeMap<NonTerminalID, StateID>> = vec![BTreeMap::new(); num_states];
    for (&state_id, row) in parser.table.iter() {
        for (nt_id, goto) in row.get_gotos() {
            if let Some(next_state) = goto.state_id {
                goto_map[state_id.0].insert(*nt_id, next_state);
            }
        }
    }

    let mut below_cache: FxHashMap<(usize, usize), Bitset> = FxHashMap::default();

    for (&state_id, row) in parser.table.iter() {
        for term_idx in 0..num_terminals {
            let terminal_id = TerminalID(term_idx);
            let action = row.get_shifts_and_reduces_for_terminal(&terminal_id);
            if let Some(action) = action {
                handle_action(
                    state_id,
                    terminal_id,
                    action,
                    num_states,
                    underneath_map,
                    &goto_map,
                    &mut below_cache,
                    &mut transitions,
                );
            }
        }
    }

    ApproximateParserNFA {
        num_states,
        start_state: parser.start_state_id,
        transitions,
    }
}

fn handle_action(
    state_id: StateID,
    terminal_id: TerminalID,
    action: Stage7ShiftsAndReducesLookaheadValue,
    num_states: usize,
    underneath_map: &[Bitset],
    goto_map: &[BTreeMap<NonTerminalID, StateID>],
    below_cache: &mut FxHashMap<(usize, usize), Bitset>,
    transitions: &mut [BTreeMap<TerminalID, Bitset>],
) {
    match action {
        Stage7ShiftsAndReducesLookaheadValue::Shift(next_state) => {
            add_nfa_transition(transitions, state_id, terminal_id, next_state, num_states);
        }
        Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
            add_reduce_transitions(
                transitions,
                state_id,
                terminal_id,
                nonterminal_id,
                len,
                num_states,
                underneath_map,
                goto_map,
                below_cache,
            );
        }
        Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
            if let Some(next_state) = shift {
                add_nfa_transition(transitions, state_id, terminal_id, next_state, num_states);
            }
            for (len, nts) in reduces {
                for (nt_id, _pids) in nts {
                    add_reduce_transitions(
                        transitions,
                        state_id,
                        terminal_id,
                        nt_id,
                        len,
                        num_states,
                        underneath_map,
                        goto_map,
                        below_cache,
                    );
                }
            }
        }
    }
}

fn add_reduce_transitions(
    transitions: &mut [BTreeMap<TerminalID, Bitset>],
    state_id: StateID,
    terminal_id: TerminalID,
    nonterminal_id: NonTerminalID,
    len: usize,
    num_states: usize,
    underneath_map: &[Bitset],
    goto_map: &[BTreeMap<NonTerminalID, StateID>],
    below_cache: &mut FxHashMap<(usize, usize), Bitset>,
) {
    let below_states = compute_states_below(
        state_id,
        len,
        num_states,
        underneath_map,
        below_cache,
    );

    for below_state in below_states.iter() {
        if let Some(&goto_state) = goto_map[below_state].get(&nonterminal_id) {
            add_nfa_transition(transitions, state_id, terminal_id, goto_state, num_states);
        }
    }
}

fn add_nfa_transition(
    transitions: &mut [BTreeMap<TerminalID, Bitset>],
    from: StateID,
    terminal: TerminalID,
    to: StateID,
    num_states: usize,
) {
    let entry = transitions[from.0]
        .entry(terminal)
        .or_insert_with(|| Bitset::new(num_states));
    entry.insert(to.0);
}

fn compute_states_below(
    start_state: StateID,
    len: usize,
    num_states: usize,
    underneath_map: &[Bitset],
    cache: &mut FxHashMap<(usize, usize), Bitset>,
) -> Bitset {
    if let Some(cached) = cache.get(&(start_state.0, len)) {
        return cached.clone();
    }

    let mut current = Bitset::new(num_states);
    current.insert(start_state.0);

    for _ in 0..len {
        let mut next = Bitset::new(num_states);
        for s in current.iter() {
            next.union_with(&underneath_map[s]);
        }
        current = next;
        if current.is_empty() {
            break;
        }
    }

    cache.insert((start_state.0, len), current.clone());
    current
}

fn determinize_nfa(nfa: &ApproximateParserNFA) -> ApproximateParserDFA {
    let mut state_map: FxHashMap<Bitset, usize> = FxHashMap::default();
    let mut dfa_state_sets: Vec<Bitset> = Vec::new();
    let mut transitions: Vec<BTreeMap<TerminalID, usize>> = Vec::new();
    let mut worklist: VecDeque<usize> = VecDeque::new();

    let mut start_set = Bitset::new(nfa.num_states);
    start_set.insert(nfa.start_state.0);
    state_map.insert(start_set.clone(), 0);
    dfa_state_sets.push(start_set);
    transitions.push(BTreeMap::new());
    worklist.push_back(0);

    while let Some(dfa_state_id) = worklist.pop_front() {
        let subset = dfa_state_sets[dfa_state_id].clone();
        let mut term_to_targets: BTreeMap<TerminalID, Bitset> = BTreeMap::new();

        for nfa_state in subset.iter() {
            for (terminal, targets) in &nfa.transitions[nfa_state] {
                term_to_targets
                    .entry(*terminal)
                    .or_insert_with(|| Bitset::new(nfa.num_states))
                    .union_with(targets);
            }
        }

        for (terminal, target_set) in term_to_targets {
            if target_set.is_empty() {
                continue;
            }
            let next_id = if let Some(&existing) = state_map.get(&target_set) {
                existing
            } else {
                let new_id = dfa_state_sets.len();
                state_map.insert(target_set.clone(), new_id);
                dfa_state_sets.push(target_set);
                transitions.push(BTreeMap::new());
                worklist.push_back(new_id);
                new_id
            };
            transitions[dfa_state_id].insert(terminal, next_id);
        }
    }

    ApproximateParserDFA {
        start_state: 0,
        transitions,
        dfa_state_sets,
    }
}
