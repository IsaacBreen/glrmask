//! Terminal characterization for template construction.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{Action, GLRTable, GuardedStackShift, StackShiftGuard};
use crate::grammar::flat::{NonterminalID, TerminalID};

type NtAdjacency = BTreeMap<NonterminalID, BTreeSet<NonterminalID>>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StackMatcher {
    Any,
    State(u32),
    States(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InitialEscape {
    pub pop: Vec<StackMatcher>,
    pub pushes: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InitialReduce {
    pub pop: Vec<StackMatcher>,
    pub nonterminal: NonterminalID,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NtEscape {
    pub source_nonterminal: NonterminalID,
    pub pop: Vec<StackMatcher>,
    pub pushes: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NtRereduce {
    pub source_nonterminal: NonterminalID,
    pub pop: Vec<StackMatcher>,
    pub target_nonterminal: NonterminalID,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TerminalCharacterization {
    pub escapes: Vec<InitialEscape>,
    pub reduces: Vec<InitialReduce>,
    pub nt_escapes: Vec<NtEscape>,
    pub nt_rereduces: Vec<NtRereduce>,
    pub all_nts: BTreeSet<NonterminalID>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RelationConfig {
    input: Vec<StackMatcher>,
    segment: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharacterizationSource {
    Initial,
    Nonterminal(NonterminalID),
}

#[derive(Debug, Default)]
struct CharacterizationOutput {
    escapes: BTreeSet<InitialEscape>,
    reduces: BTreeSet<InitialReduce>,
    nt_escapes: BTreeSet<NtEscape>,
    nt_rereduces: BTreeSet<NtRereduce>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StackEdit {
    pop: Vec<StackMatcher>,
    pushes: Vec<u32>,
}

impl CharacterizationOutput {
    fn emit_escape(
        &mut self,
        source: CharacterizationSource,
        pop: Vec<StackMatcher>,
        pushes: Vec<u32>,
    ) {
        match source {
            CharacterizationSource::Initial => {
                self.escapes.insert(InitialEscape { pop, pushes });
            }
            CharacterizationSource::Nonterminal(source_nonterminal) => {
                self.nt_escapes.insert(NtEscape {
                    source_nonterminal,
                    pop,
                    pushes,
                });
            }
        }
    }

    fn emit_reduce(
        &mut self,
        source: CharacterizationSource,
        pop: Vec<StackMatcher>,
        nonterminal: NonterminalID,
    ) {
        debug_assert!(
            !pop.is_empty(),
            "terminal characterization should not emit zero-length reduce paths"
        );

        match source {
            CharacterizationSource::Initial => {
                self.reduces.insert(InitialReduce { pop, nonterminal });
            }
            CharacterizationSource::Nonterminal(source_nonterminal) => {
                self.nt_rereduces.insert(NtRereduce {
                    source_nonterminal,
                    pop,
                    target_nonterminal: nonterminal,
                });
            }
        }
    }
}

impl TerminalCharacterization {
    pub fn find_cycle(&self) -> Option<Vec<NonterminalID>> {
        find_nonterminal_cycle(&build_rereduce_adjacency(&self.nt_rereduces))
    }
}

fn build_rereduce_adjacency(nt_rereduces: &[NtRereduce]) -> NtAdjacency {
    let mut adjacency = NtAdjacency::new();
    for rereduce in nt_rereduces {
        adjacency
            .entry(rereduce.source_nonterminal)
            .or_default()
            .insert(rereduce.target_nonterminal);
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

fn sorted_dedup(mut states: Vec<u32>) -> Vec<u32> {
    states.sort_unstable();
    states.dedup();
    states
}

fn states_matcher(states: Vec<u32>) -> Option<StackMatcher> {
    let states = sorted_dedup(states);
    match states.len() {
        0 => None,
        1 => Some(StackMatcher::State(states[0])),
        _ => Some(StackMatcher::States(states)),
    }
}

fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    out
}

fn constrain_matcher(matcher: &mut StackMatcher, allowed: &[u32]) -> bool {
    let allowed = sorted_dedup(allowed.to_vec());

    let next = match matcher {
        StackMatcher::Any => states_matcher(allowed),
        StackMatcher::State(state) => {
            if allowed.binary_search(state).is_ok() {
                Some(StackMatcher::State(*state))
            } else {
                None
            }
        }
        StackMatcher::States(states) => states_matcher(intersect_sorted(states, &allowed)),
    };

    if let Some(next) = next {
        *matcher = next;
        true
    } else {
        false
    }
}

fn finite_states(matcher: &StackMatcher) -> Option<Vec<u32>> {
    match matcher {
        StackMatcher::Any => None,
        StackMatcher::State(state) => Some(vec![*state]),
        StackMatcher::States(states) => Some(states.clone()),
    }
}

fn assert_well_formed_guarded_shift(shift: &GuardedStackShift) {
    debug_assert!(
        shift.guards.windows(2).all(|w| w[0].pop <= w[1].pop),
        "GuardedStackShift guards must be sorted by pop"
    );
    debug_assert!(
        shift.guards.iter().all(|guard| guard.pop <= shift.pop),
        "GuardedStackShift guard.pop must be <= shift.pop"
    );
    debug_assert!(
        shift.guards.iter().all(|guard| !guard.states.is_empty()),
        "GuardedStackShift guards must have nonempty state sets"
    );
    debug_assert!(
        shift.guards.iter().all(|guard| guard.states.windows(2).all(|w| w[0] < w[1])),
        "GuardedStackShift guard.states must be sorted and deduplicated"
    );
}

fn identity_config(top_state: u32) -> RelationConfig {
    RelationConfig {
        input: vec![StackMatcher::State(top_state)],
        segment: vec![top_state],
    }
}

fn start_relation_after_goto(
    revealed_state: u32,
    goto_state: u32,
    goto_replace: bool,
) -> RelationConfig {
    RelationConfig {
        input: vec![StackMatcher::State(revealed_state)],
        segment: if goto_replace {
            vec![goto_state]
        } else {
            vec![revealed_state, goto_state]
        },
    }
}

fn apply_goto_to_relation_config(
    config: &RelationConfig,
    reveal_index: usize,
    goto_state: u32,
    goto_replace: bool,
) -> RelationConfig {
    let mut segment = config.segment[..=reveal_index].to_vec();
    if goto_replace {
        segment.pop();
    }
    segment.push(goto_state);

    RelationConfig {
        input: config.input.clone(),
        segment,
    }
}

fn reduce_crossing_pattern(config: &RelationConfig, reduce_len: usize) -> Vec<StackMatcher> {
    debug_assert!(reduce_len >= config.segment.len());

    let mut pop = config.input.clone();
    pop.extend(std::iter::repeat(StackMatcher::Any).take(reduce_len - config.segment.len()));
    pop
}

fn ensure_input_len(input: &mut Vec<StackMatcher>, len: usize) {
    while input.len() < len {
        input.push(StackMatcher::Any);
    }
}

fn guard_constraints(
    shift_pop: usize,
    guards: &[StackShiftGuard],
) -> Option<BTreeMap<usize, Vec<u32>>> {
    let mut out: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut previous_pop = None;

    for guard in guards {
        let depth = guard.pop as usize;

        if let Some(previous) = previous_pop {
            if depth < previous {
                return None;
            }
        }
        previous_pop = Some(depth);

        if depth > shift_pop {
            return None;
        }

        let states = sorted_dedup(guard.states.clone());
        if states.is_empty() {
            return None;
        }

        out.entry(depth)
            .and_modify(|existing| {
                *existing = intersect_sorted(existing, &states);
            })
            .or_insert(states);

        if out.get(&depth).is_some_and(|states| states.is_empty()) {
            return None;
        }
    }

    Some(out)
}

fn stack_effect_edits(
    config: &RelationConfig,
    shift_pop: usize,
    shifted_pushes: &[u32],
    guards: &[StackShiftGuard],
) -> Vec<StackEdit> {
    let Some(guards) = guard_constraints(shift_pop, guards) else {
        return Vec::new();
    };

    let segment_len = config.segment.len();
    let base_input_len = config.input.len();
    let mut input = config.input.clone();

    for (&depth, allowed) in &guards {
        if depth < segment_len {
            let state = config.segment[segment_len - 1 - depth];
            if allowed.binary_search(&state).is_err() {
                return Vec::new();
            }
        } else {
            let below_segment_index = depth - segment_len;
            let input_index = base_input_len + below_segment_index;
            ensure_input_len(&mut input, input_index + 1);
            if !constrain_matcher(&mut input[input_index], allowed) {
                return Vec::new();
            }
        }
    }

    let unknown_popped = shift_pop.saturating_sub(segment_len);
    ensure_input_len(&mut input, base_input_len + unknown_popped);

    if shift_pop < segment_len {
        let mut pushes = config.segment[..segment_len - shift_pop].to_vec();
        pushes.extend_from_slice(shifted_pushes);
        return vec![StackEdit { pop: input, pushes }];
    }

    if guards.contains_key(&shift_pop) {
        let revealed_input_index = base_input_len + unknown_popped;
        ensure_input_len(&mut input, revealed_input_index + 1);
        let Some(revealed_states) = finite_states(&input[revealed_input_index]) else {
            unreachable!("revealed guard should have constrained matcher to a finite state set");
        };

        let mut edits = Vec::new();
        for revealed_state in revealed_states {
            let mut branch_input = input.clone();
            branch_input[revealed_input_index] = StackMatcher::State(revealed_state);
            let mut pushes = Vec::with_capacity(1 + shifted_pushes.len());
            pushes.push(revealed_state);
            pushes.extend_from_slice(shifted_pushes);
            edits.push(StackEdit {
                pop: branch_input,
                pushes,
            });
        }
        edits
    } else {
        vec![StackEdit {
            pop: input,
            pushes: shifted_pushes.to_vec(),
        }]
    }
}

fn process_reduce_from_config(
    table: &GLRTable,
    source: CharacterizationSource,
    config: &RelationConfig,
    lhs: NonterminalID,
    reduce_len: usize,
    output: &mut CharacterizationOutput,
    seen: &mut BTreeSet<RelationConfig>,
    worklist: &mut VecDeque<RelationConfig>,
) {
    if reduce_len == 0 {
        let Some(&top_state) = config.segment.last() else {
            return;
        };

        let Some((goto_state, goto_replace)) = table.goto_target(top_state, lhs) else {
            return;
        };

        let next = apply_goto_to_relation_config(
            config,
            config.segment.len() - 1,
            goto_state,
            goto_replace,
        );
        if seen.insert(next.clone()) {
            worklist.push_back(next);
        }
        return;
    }

    if reduce_len < config.segment.len() {
        let reveal_index = config.segment.len() - reduce_len - 1;
        let revealed_state = config.segment[reveal_index];
        let Some((goto_state, goto_replace)) = table.goto_target(revealed_state, lhs) else {
            return;
        };

        let next = apply_goto_to_relation_config(config, reveal_index, goto_state, goto_replace);
        if seen.insert(next.clone()) {
            worklist.push_back(next);
        }
        return;
    }

    output.emit_reduce(source, reduce_crossing_pattern(config, reduce_len), lhs);
}

fn emit_stack_effect_from_config(
    source: CharacterizationSource,
    config: &RelationConfig,
    shift_pop: usize,
    shifted_pushes: &[u32],
    guards: &[StackShiftGuard],
    output: &mut CharacterizationOutput,
) {
    for edit in stack_effect_edits(config, shift_pop, shifted_pushes, guards) {
        output.emit_escape(source, edit.pop, edit.pushes);
    }
}

fn process_action_from_config(
    table: &GLRTable,
    source: CharacterizationSource,
    config: &RelationConfig,
    action: &Action,
    reduce_len_adjustment: usize,
    output: &mut CharacterizationOutput,
    seen: &mut BTreeSet<RelationConfig>,
    worklist: &mut VecDeque<RelationConfig>,
) {
    match action {
        Action::Shift(shift_state, replace) => {
            if *replace {
                emit_stack_effect_from_config(source, config, 1, &[*shift_state], &[], output);
            } else {
                emit_stack_effect_from_config(source, config, 0, &[*shift_state], &[], output);
            }
        }
        Action::StackShifts(shifts) => {
            for shift in shifts {
                emit_stack_effect_from_config(
                    source,
                    config,
                    shift.pop as usize,
                    &shift.pushes,
                    &[],
                    output,
                );
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                assert_well_formed_guarded_shift(shift);
                emit_stack_effect_from_config(
                    source,
                    config,
                    shift.pop as usize,
                    &shift.pushes,
                    &shift.guards,
                    output,
                );
            }
        }
        Action::Reduce(lhs, len) => {
            process_reduce_from_config(
                table,
                source,
                config,
                *lhs,
                (*len as usize) + reduce_len_adjustment,
                output,
                seen,
                worklist,
            );
        }
        Action::Split {
            shift,
            reduces,
            accept: _,
        } => {
            if let Some((shift_state, replace)) = shift {
                if *replace {
                    emit_stack_effect_from_config(source, config, 1, &[*shift_state], &[], output);
                } else {
                    emit_stack_effect_from_config(source, config, 0, &[*shift_state], &[], output);
                }
            }

            for &(lhs, len) in reduces {
                process_reduce_from_config(
                    table,
                    source,
                    config,
                    lhs,
                    (len as usize) + reduce_len_adjustment,
                    output,
                    seen,
                    worklist,
                );
            }
        }
        Action::Accept => {}
    }
}

fn drain_nonconsuming_worklist(
    table: &GLRTable,
    terminal: TerminalID,
    source: CharacterizationSource,
    output: &mut CharacterizationOutput,
    seen: &mut BTreeSet<RelationConfig>,
    worklist: &mut VecDeque<RelationConfig>,
) {
    while let Some(config) = worklist.pop_front() {
        let Some(&top_state) = config.segment.last() else {
            continue;
        };
        let Some(action) = table.action(top_state, terminal) else {
            continue;
        };

        process_action_from_config(table, source, &config, action, 0, output, seen, worklist);
    }
}

fn characterize_initial_actions_for_terminal(
    table: &GLRTable,
    terminal: TerminalID,
    output: &mut CharacterizationOutput,
) {
    for state in 0..table.num_states {
        let Some(action) = table.action(state, terminal) else {
            continue;
        };

        let config = identity_config(state);
        let mut seen = BTreeSet::from([config.clone()]);
        let mut worklist = VecDeque::new();
        let reduce_len_adjustment = usize::from(table.forwarded_shifts.contains(&(state, terminal)));

        process_action_from_config(
            table,
            CharacterizationSource::Initial,
            &config,
            action,
            reduce_len_adjustment,
            output,
            &mut seen,
            &mut worklist,
        );

        drain_nonconsuming_worklist(
            table,
            terminal,
            CharacterizationSource::Initial,
            output,
            &mut seen,
            &mut worklist,
        );
    }
}

fn characterize_nt_continuations_for_terminal(
    table: &GLRTable,
    terminal: TerminalID,
    output: &mut CharacterizationOutput,
) {
    for revealed_state in 0..table.num_states {
        let Some(gotos) = table.goto.get(revealed_state as usize) else {
            continue;
        };

        for (&nonterminal, &(goto_state, goto_replace)) in gotos {
            let config = start_relation_after_goto(revealed_state, goto_state, goto_replace);
            let mut seen = BTreeSet::from([config.clone()]);
            let mut worklist = VecDeque::from([config]);

            drain_nonconsuming_worklist(
                table,
                terminal,
                CharacterizationSource::Nonterminal(nonterminal),
                output,
                &mut seen,
                &mut worklist,
            );
        }
    }
}

fn characterize_terminal(
    table: &GLRTable,
    terminal: TerminalID,
) -> TerminalCharacterization {
    let mut output = CharacterizationOutput::default();
    characterize_initial_actions_for_terminal(table, terminal, &mut output);
    characterize_nt_continuations_for_terminal(table, terminal, &mut output);

    let mut referenced_nts = BTreeSet::new();
    for reduce in &output.reduces {
        referenced_nts.insert(reduce.nonterminal);
    }
    for nt_escape in &output.nt_escapes {
        referenced_nts.insert(nt_escape.source_nonterminal);
    }
    for nt_rereduce in &output.nt_rereduces {
        referenced_nts.insert(nt_rereduce.source_nonterminal);
        referenced_nts.insert(nt_rereduce.target_nonterminal);
    }

    let characterization = TerminalCharacterization {
        escapes: output.escapes.into_iter().collect(),
        reduces: output.reduces.into_iter().collect(),
        nt_escapes: output.nt_escapes.into_iter().collect(),
        nt_rereduces: output.nt_rereduces.into_iter().collect(),
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

pub(crate) fn characterize_terminals(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
) -> BTreeMap<TerminalID, TerminalCharacterization> {
    let debug = std::env::var("GLRMASK_DEBUG_PROFILE").is_ok();

    if debug {
        let total_gotos: usize = table.goto.iter().map(|g| g.len()).sum();
        eprintln!(
            "[glrmask/debug][characterize] glr_states={} total_gotos={} num_terminals={}",
            table.num_states, total_gotos, grammar.num_terminals
        );
    }

    (0..grammar.num_terminals)
        .map(|terminal| {
            let t0 = std::time::Instant::now();
            let characterization = characterize_terminal(table, terminal);
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

#[cfg(test)]
mod tests {
    use super::*;

    use rustc_hash::FxHashSet;

    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::table::{ActionRow, GotoRow, GLRTable};
    use crate::grammar::flat::Rule;

    fn empty_table(num_states: u32, num_terminals: u32) -> GLRTable {
        GLRTable {
            action: vec![ActionRow::default(); num_states as usize],
            goto: vec![GotoRow::default(); num_states as usize],
            num_states,
            num_terminals,
            num_rules: 0,
            rules: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
        }
    }

    fn test_grammar() -> AnalyzedGrammar {
        AnalyzedGrammar {
            rules: Vec::<Rule>::new(),
            start: 0,
            num_terminals: 1,
            terminal_display_names: vec!["t0".to_string()],
            num_nonterminals: 64,
            nullable: BTreeSet::new(),
            first: vec![BTreeSet::new(); 64],
            follow: vec![BTreeSet::new(); 64],
            rules_by_lhs: vec![Vec::new(); 64],
        }
    }

    fn characterize_one(table: &GLRTable, terminal: TerminalID) -> TerminalCharacterization {
        characterize_terminals(table, &test_grammar())
            .remove(&terminal)
            .expect("terminal characterization should exist")
    }

    fn state(state: u32) -> StackMatcher {
        StackMatcher::State(state)
    }

    #[test]
    fn test_initial_zero_pop_replace_goto_preserves_replacement_state() {
        let terminal = 0;
        let reduce_nt = 10;
        let mut table = empty_table(4, 1);
        table.action[1].insert(terminal, Action::Reduce(reduce_nt, 0));
        table.goto[1].insert(reduce_nt, (2, true));
        table.action[2].insert(terminal, Action::Shift(3, false));

        let characterization = characterize_one(&table, terminal);
        assert!(characterization.reduces.is_empty());
        assert!(characterization.escapes.contains(&InitialEscape {
            pop: vec![state(1)],
            pushes: vec![2, 3],
        }));
        assert!(!characterization.escapes.contains(&InitialEscape {
            pop: vec![state(1)],
            pushes: vec![1, 3],
        }));
    }

    #[test]
    fn test_goto_zero_pop_chain_preserves_replaced_segment_geometry() {
        let terminal = 0;
        let stack_nt = 20;
        let reduce_b = 31;
        let mut table = empty_table(5, 1);

        table.goto[0].insert(stack_nt, (1, true));
        table.action[1].insert(terminal, Action::Reduce(reduce_b, 0));
        table.goto[1].insert(reduce_b, (2, false));
        table.action[2].insert(terminal, Action::Shift(3, false));

        let characterization = characterize_one(&table, terminal);
        assert!(characterization.nt_rereduces.is_empty());
        assert!(characterization.nt_escapes.contains(&NtEscape {
            source_nonterminal: stack_nt,
            pop: vec![state(0)],
            pushes: vec![1, 2, 3],
        }));
    }

    #[test]
    fn test_guard_inside_popped_region_constrains_escape() {
        let terminal = 0;
        let mut table = empty_table(64, 1);
        table.action[10].insert(
            terminal,
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![3],
                }],
                pop: 2,
                pushes: vec![40],
            }]),
        );

        let characterization = characterize_one(&table, terminal);
        assert!(characterization.escapes.contains(&InitialEscape {
            pop: vec![state(10), state(3)],
            pushes: vec![40],
        }));
        assert!(!characterization.escapes.contains(&InitialEscape {
            pop: vec![state(10), StackMatcher::Any],
            pushes: vec![40],
        }));
    }

    #[test]
    fn test_guard_at_revealed_state_is_checked_and_preserved() {
        let terminal = 0;
        let mut table = empty_table(64, 1);
        table.action[10].insert(
            terminal,
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![3],
                }],
                pop: 1,
                pushes: vec![40],
            }]),
        );

        let characterization = characterize_one(&table, terminal);
        assert!(characterization.escapes.contains(&InitialEscape {
            pop: vec![state(10), state(3)],
            pushes: vec![3, 40],
        }));
    }

    #[test]
    fn test_guarded_revealed_state_branches_exactly() {
        let terminal = 0;
        let mut table = empty_table(64, 1);
        table.action[10].insert(
            terminal,
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![3, 7],
                }],
                pop: 1,
                pushes: vec![40],
            }]),
        );

        let characterization = characterize_one(&table, terminal);
        assert!(characterization.escapes.contains(&InitialEscape {
            pop: vec![state(10), state(3)],
            pushes: vec![3, 40],
        }));
        assert!(characterization.escapes.contains(&InitialEscape {
            pop: vec![state(10), state(7)],
            pushes: vec![7, 40],
        }));
        assert_eq!(characterization.escapes.len(), 2);
    }

    #[test]
    fn test_duplicate_guard_depths_intersect() {
        let terminal = 0;
        let mut table = empty_table(64, 1);
        table.action[10].insert(
            terminal,
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![
                    StackShiftGuard {
                        pop: 1,
                        states: vec![3, 7],
                    },
                    StackShiftGuard {
                        pop: 1,
                        states: vec![7, 9],
                    },
                ],
                pop: 1,
                pushes: vec![40],
            }]),
        );

        let characterization = characterize_one(&table, terminal);
        assert_eq!(
            characterization.escapes,
            vec![InitialEscape {
                pop: vec![state(10), state(7)],
                pushes: vec![7, 40],
            }]
        );
    }

    #[test]
    fn test_guard_after_zero_pop_goto_uses_preserved_segment_state() {
        let terminal = 0;
        let reduce_nt = 10;
        let mut table = empty_table(8, 1);

        table.action[1].insert(terminal, Action::Reduce(reduce_nt, 0));
        table.goto[1].insert(reduce_nt, (2, false));
        table.action[2].insert(
            terminal,
            Action::GuardedStackShifts(vec![GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![1],
                }],
                pop: 1,
                pushes: vec![3],
            }]),
        );

        let characterization = characterize_one(&table, terminal);
        assert!(characterization.escapes.contains(&InitialEscape {
            pop: vec![state(1)],
            pushes: vec![1, 3],
        }));
    }
}
