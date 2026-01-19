use crate::datastructures::bitset::Bitset;
use crate::glr::parser::GLRParser;
use crate::glr::table::{Stage7ShiftsAndReducesLookaheadValue, StateID, Table, TerminalID, NonTerminalID};
use rustc_hash::FxHashMap;
use rustfst::algorithms::determinize::{determinize_with_config, DeterminizeConfig, DeterminizeType};
use rustfst::algorithms::minimize_with_config;
use rustfst::algorithms::rm_epsilon::rm_epsilon;
use rustfst::fst_traits::CoreFst;
use rustfst::prelude::{ExpandedFst, MinimizeConfig, MutableFst, StateId, Tr, Trs, VectorFst, EPS_LABEL};
use rustfst::semirings::TropicalWeight;
use rustfst::Semiring;
use std::collections::BTreeMap;

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

const NO_STATE: usize = usize::MAX;

#[derive(Debug, Clone)]
pub struct LazyApproximateDFA {
    nfa: ApproximateParserNFA,
    state_cache: FxHashMap<Bitset, usize>,
    transitions: Vec<BTreeMap<TerminalID, usize>>,
    dfa_state_sets: Vec<Bitset>,
    pub start_state: usize,
}

impl LazyApproximateDFA {
    pub fn new(nfa: ApproximateParserNFA) -> Self {
        let mut initial_set = Bitset::new(nfa.num_states);
        if nfa.num_states > 0 {
            initial_set.insert(nfa.start_state.0);
        }
        Self::new_with_initial(nfa, initial_set)
    }

    pub fn new_with_initial(nfa: ApproximateParserNFA, initial_set: Bitset) -> Self {
        let mut state_cache = FxHashMap::default();
        state_cache.insert(initial_set.clone(), 0);

        Self {
            nfa,
            state_cache,
            transitions: vec![BTreeMap::new()],
            dfa_state_sets: vec![initial_set],
            start_state: 0,
        }
    }

    pub fn step(&mut self, state: usize, terminal: TerminalID) -> Option<usize> {
        if let Some(&next) = self.transitions[state].get(&terminal) {
            return if next == NO_STATE { None } else { Some(next) };
        }

        let current_set = &self.dfa_state_sets[state];
        let mut next_set = Bitset::new(self.nfa.num_states);

        for nfa_state in current_set.iter() {
            if let Some(targets) = self.nfa.transitions[nfa_state].get(&terminal) {
                next_set.union_with(targets);
            }
        }

        if next_set.is_empty() {
            self.transitions[state].insert(terminal, NO_STATE);
            return None;
        }

        let next_dfa_state = if let Some(&existing) = self.state_cache.get(&next_set) {
            existing
        } else {
            let new_id = self.dfa_state_sets.len();
            self.state_cache.insert(next_set.clone(), new_id);
            self.dfa_state_sets.push(next_set);
            self.transitions.push(BTreeMap::new());
            new_id
        };

        self.transitions[state].insert(terminal, next_dfa_state);
        Some(next_dfa_state)
    }
}

#[derive(Default)]
struct ReduceStats {
    counts: Vec<usize>,
    size_sum: Vec<usize>,
    size_max: Vec<usize>,
}

impl ReduceStats {
    fn record(&mut self, len: usize, size: usize) {
        if len >= self.counts.len() {
            let new_len = len + 1;
            self.counts.resize(new_len, 0);
            self.size_sum.resize(new_len, 0);
            self.size_max.resize(new_len, 0);
        }
        self.counts[len] += 1;
        self.size_sum[len] += size;
        self.size_max[len] = self.size_max[len].max(size);
    }
}

pub fn build_approximate_parser_dfa(parser: &GLRParser) -> LazyApproximateDFA {
    let num_states = table_state_count(&parser.table);
    crate::debug!(4, "Approximate DFA: building lazy DFA from parser with {} states", num_states);
    let underneath_map = compute_underneath_map(&parser.table, num_states);
    let nfa = build_nfa(parser, num_states, &underneath_map);
    if crate::r#macro::is_debug_level_enabled(5) {
        let mut total_edges = 0usize;
        for map in &nfa.transitions {
            for targets in map.values() {
                total_edges += targets.len();
            }
        }
        let avg = if nfa.num_states == 0 { 0.0 } else { total_edges as f64 / nfa.num_states as f64 };
        crate::debug!(5, "Approximate DFA: NFA has {} transitions (avg {:.2} per state)", total_edges, avg);
    }
    let initial_set = Bitset::ones(nfa.num_states);
    LazyApproximateDFA::new_with_initial(nfa, initial_set)
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
    let debug_enabled = crate::r#macro::is_debug_level_enabled(5);
    let mut reduce_stats = ReduceStats::default();

    for (&state_id, row) in parser.table.iter() {
        for term_idx in 0..num_terminals {
            let terminal_id = TerminalID(term_idx);
            let action = row.get_shifts_and_reduces_for_terminal(&terminal_id);
            if let Some(action) = action {
                if debug_enabled {
                    handle_action(
                        state_id,
                        terminal_id,
                        action,
                        num_states,
                        underneath_map,
                        &goto_map,
                        &mut below_cache,
                        &mut transitions,
                        Some(&mut reduce_stats),
                    );
                } else {
                    handle_action(
                        state_id,
                        terminal_id,
                        action,
                        num_states,
                        underneath_map,
                        &goto_map,
                        &mut below_cache,
                        &mut transitions,
                        None,
                    );
                }
            }
        }
    }

    if crate::r#macro::is_debug_level_enabled(5) {
        let mut total_labels = 0usize;
        let mut total_targets = 0usize;
        let mut max_labels_state = 0usize;
        let mut max_targets_label = 0usize;
        for map in &transitions {
            total_labels += map.len();
            max_labels_state = max_labels_state.max(map.len());
            for targets in map.values() {
                let len = targets.len();
                total_targets += len;
                max_targets_label = max_targets_label.max(len);
            }
        }
        let avg_labels = if num_states == 0 { 0.0 } else { total_labels as f64 / num_states as f64 };
        let avg_targets = if total_labels == 0 { 0.0 } else { total_targets as f64 / total_labels as f64 };
        crate::debug!(5, "Approximate DFA NFA stats: states={}, labels={}, avg_labels_per_state={:.2}, avg_targets_per_label={:.2}, max_labels_state={}, max_targets_label={}",
            num_states, total_labels, avg_labels, avg_targets, max_labels_state, max_targets_label);
        crate::debug!(5, "Approximate DFA NFA reduce-cache entries: {}", below_cache.len());

        if num_terminals > 0 {
            let mut per_term_counts = vec![0usize; num_terminals];
            let mut per_term_target_sum = vec![0usize; num_terminals];
            let mut per_term_target_max = vec![0usize; num_terminals];
            for map in &transitions {
                for (terminal, targets) in map {
                    let idx = terminal.0;
                    if idx >= num_terminals {
                        continue;
                    }
                    per_term_counts[idx] += 1;
                    let len = targets.len();
                    per_term_target_sum[idx] += len;
                    per_term_target_max[idx] = per_term_target_max[idx].max(len);
                }
            }

            let mut term_indices: Vec<usize> = (0..num_terminals).collect();
            term_indices.sort_by_key(|&i| std::cmp::Reverse(per_term_target_max[i]));
            crate::debug!(5, "Approximate DFA NFA: top terminals by target set size:");
            for idx in term_indices.into_iter().take(10) {
                let count = per_term_counts[idx];
                if count == 0 {
                    continue;
                }
                let avg = per_term_target_sum[idx] as f64 / count as f64;
                let name = parser.terminal_map.get_by_right(&TerminalID(idx))
                    .map(|t| format!("{}", t))
                    .unwrap_or_else(|| format!("T{}", idx));
                let ignored = parser.ignore_terminal_ids.contains(&TerminalID(idx));
                crate::debug!(5, "  tid={} ({}) ignored={} transitions={}, avg_targets={:.2}, max_targets={}", idx, name, ignored, count, avg, per_term_target_max[idx]);
            }

            for tid in [43usize, 44usize] {
                if tid >= num_terminals {
                    continue;
                }
                let name = parser.terminal_map.get_by_right(&TerminalID(tid))
                    .map(|t| format!("{}", t))
                    .unwrap_or_else(|| format!("T{}", tid));
                let ignored = parser.ignore_terminal_ids.contains(&TerminalID(tid));
                crate::debug!(5, "Approximate DFA NFA: tid={} name={} ignored={} transitions={} max_targets={}",
                    tid, name, ignored, per_term_counts[tid], per_term_target_max[tid]);

                let mut sample_targets: Vec<usize> = Vec::new();
                for map in &transitions {
                    if let Some(targets) = map.get(&TerminalID(tid)) {
                        for target in targets.iter() {
                            sample_targets.push(target);
                            if sample_targets.len() >= 20 {
                                break;
                            }
                        }
                    }
                    if sample_targets.len() >= 20 {
                        break;
                    }
                }
                crate::debug!(5, "Approximate DFA NFA: tid={} sample targets (first 20)={:?}", tid, sample_targets);
            }
        }

        if !reduce_stats.counts.is_empty() {
            let mut len_indices: Vec<usize> = (0..reduce_stats.counts.len()).collect();
            len_indices.sort_by_key(|&len| std::cmp::Reverse(reduce_stats.size_max[len]));
            crate::debug!(5, "Approximate DFA NFA: top reduce lengths by below-set size:");
            for len in len_indices.into_iter().take(10) {
                let count = reduce_stats.counts[len];
                if count == 0 {
                    continue;
                }
                let avg = reduce_stats.size_sum[len] as f64 / count as f64;
                crate::debug!(5, "  len={} count={}, avg_below_size={:.2}, max_below_size={}", len, count, avg, reduce_stats.size_max[len]);
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
    mut reduce_stats: Option<&mut ReduceStats>,
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
                reduce_stats.as_deref_mut(),
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
                        reduce_stats.as_deref_mut(),
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
    mut reduce_stats: Option<&mut ReduceStats>,
) {
    let below_states = compute_states_below(
        state_id,
        len,
        num_states,
        underneath_map,
        below_cache,
    );

    if let Some(stats) = reduce_stats.as_deref_mut() {
        stats.record(len, below_states.len());
    }

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

#[inline]
fn terminal_id_to_fst_label(terminal: TerminalID) -> u32 {
    let base = u32::try_from(terminal.0).expect("terminal id exceeds fst label range");
    let label = base.checked_add(1).expect("terminal id too large for fst label");
    debug_assert_ne!(label, EPS_LABEL);
    label
}

#[inline]
fn fst_label_to_terminal_id(label: u32) -> TerminalID {
    assert_ne!(label, EPS_LABEL);
    TerminalID(usize::try_from(label - 1).expect("fst label is not a valid terminal id"))
}

fn determinize_nfa(nfa: &ApproximateParserNFA) -> ApproximateParserDFA {
    if nfa.num_states == 0 {
        return ApproximateParserDFA {
            start_state: 0,
            transitions: Vec::new(),
            dfa_state_sets: Vec::new(),
        };
    }

    let mut fst: VectorFst<TropicalWeight> = VectorFst::new();
    let state_map: Vec<StateId> = (0..nfa.num_states)
        .map(|_| fst.add_state())
        .collect();

    let start_state = state_map[nfa.start_state.0];
    fst.set_start(start_state).unwrap();

    for &state in &state_map {
        fst.set_final(state, TropicalWeight::one()).unwrap();
    }

    for (from_state, map) in nfa.transitions.iter().enumerate() {
        let fst_src = state_map[from_state];
        for (terminal, targets) in map {
            let label = terminal_id_to_fst_label(*terminal);
            for to_state in targets.iter() {
                fst.add_tr(fst_src, Tr::new(label, label, TropicalWeight::one(), state_map[to_state])).unwrap();
            }
        }
    }

    rm_epsilon(&mut fst).unwrap();
    let nfa_min_config = MinimizeConfig::default().with_allow_nondet(true);
    minimize_with_config(&mut fst, nfa_min_config).unwrap();
    let start = std::time::Instant::now();
    let config = DeterminizeConfig::default().with_det_type(DeterminizeType::DeterminizeFunctional);
    let mut det_fst: VectorFst<TropicalWeight> = determinize_with_config(&fst, config).unwrap();
    minimize_with_config(&mut det_fst, MinimizeConfig::default()).unwrap();

    let mut transitions = Vec::with_capacity(det_fst.num_states());
    let mut trans_count = 0usize;
    for fst_state_id in 0..det_fst.num_states() {
        let mut map = BTreeMap::new();
        for tr in det_fst.get_trs(fst_state_id as StateId).unwrap().trs() {
            if tr.ilabel == EPS_LABEL {
                continue;
            }
            let terminal = fst_label_to_terminal_id(tr.ilabel);
            map.insert(terminal, tr.nextstate as usize);
            trans_count += 1;
        }
        transitions.push(map);
    }

    crate::debug!(
        4,
        "Approximate DFA: rustfst determinize+minimize -> {} states, {} trans in {:?}",
        transitions.len(),
        trans_count,
        start.elapsed()
    );

    let start_state = det_fst.start().map(|s| s as usize).unwrap_or(0);

    ApproximateParserDFA {
        start_state,
        transitions,
        dfa_state_sets: Vec::new(),
    }
}
