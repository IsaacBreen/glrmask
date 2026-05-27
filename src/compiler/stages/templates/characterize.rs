//! Terminal characterization for template construction.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
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

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TerminalCharacterizationProfile {
    pub(crate) terminals: usize,
    pub(crate) unique_action_signatures: usize,
    pub(crate) max_action_signature_multiplicity: usize,
    pub(crate) quotient_hits: usize,
    pub(crate) signature_ms: f64,
    pub(crate) characterize_ms: f64,
    pub(crate) fanout_ms: f64,
    pub(crate) validation_ms: f64,
    pub(crate) total_ms: f64,
    pub(crate) quotient_disabled: bool,
}

type DenseTerminalActionSignature = Vec<(u32, Option<Action>, bool)>;
type TerminalActionSignature<'a> = Vec<(u32, Option<&'a Action>, bool)>;
const NT_CONTINUATION_PARALLEL_THRESHOLD: usize = 256;
const NT_CONTINUATION_CHUNK_SIZE: usize = 64;

#[derive(Debug, Clone, Default)]
struct CharacterizationIndex {
    action_states_by_terminal: Vec<Vec<u32>>,
    forwarded_only_states_by_terminal: Vec<Vec<u32>>,
    goto_predecessors_by_target: Vec<Vec<(u32, NonterminalID, bool)>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    fn extend_from(&mut self, other: CharacterizationOutput) {
        self.escapes.extend(other.escapes);
        self.reduces.extend(other.reduces);
        self.nt_escapes.extend(other.nt_escapes);
        self.nt_rereduces.extend(other.nt_rereduces);
    }

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

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn characterization_quotient_disabled() -> bool {
    env_flag_enabled("GLRMASK_DISABLE_CHARACTERIZATION_QUOTIENT")
}

fn characterization_quotient_validation_enabled() -> bool {
    env_flag_enabled("GLRMASK_VALIDATE_CHARACTERIZATION_QUOTIENT")
}

fn sparse_action_signature_validation_enabled() -> bool {
    env_flag_enabled("GLRMASK_VALIDATE_SPARSE_ACTION_SIGNATURES")
}

fn dense_terminal_action_signature(
    table: &GLRTable,
    terminal: TerminalID,
) -> DenseTerminalActionSignature {
    let mut signature = Vec::new();
    for state in 0..table.num_states {
        let forwarded = table.forwarded_shifts.contains(&(state, terminal));
        if let Some(action) = table.action(state, terminal) {
            signature.push((state, Some(action.clone()), forwarded));
        } else if forwarded {
            // This should not occur for a well-formed table, but including the
            // forwarded bit keeps the quotient conservative if construction
            // bugs or future table encodings violate that expectation.
            signature.push((state, None, true));
        }
    }
    signature
}

fn terminal_action_signature_from_index<'a>(
    table: &'a GLRTable,
    index: &CharacterizationIndex,
    terminal: TerminalID,
) -> TerminalActionSignature<'a> {
    let action_states = index
        .action_states_by_terminal
        .get(terminal as usize)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let forwarded_only_states = index
        .forwarded_only_states_by_terminal
        .get(terminal as usize)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    let mut signature = Vec::with_capacity(action_states.len() + forwarded_only_states.len());
    for &state in action_states {
        let action = table.action(state, terminal).unwrap_or_else(|| {
            panic!("terminal action index referenced missing action ({state}, {terminal})")
        });
        let forwarded = table.forwarded_shifts.contains(&(state, terminal));
        signature.push((state, Some(action), forwarded));
    }

    for &state in forwarded_only_states {
        signature.push((state, None, true));
    }

    if !forwarded_only_states.is_empty() {
        signature.sort_by_key(|(state, _, _)| *state);
    }

    signature
}

fn owned_terminal_action_signature_from_index(
    table: &GLRTable,
    index: &CharacterizationIndex,
    terminal: TerminalID,
) -> DenseTerminalActionSignature {
    terminal_action_signature_from_index(table, index, terminal)
        .into_iter()
        .map(|(state, action, forwarded)| (state, action.cloned(), forwarded))
        .collect()
}

fn group_terminals_by_action_signature<'a>(
    table: &'a GLRTable,
    index: &CharacterizationIndex,
    grammar: &AnalyzedGrammar,
) -> (Vec<Vec<TerminalID>>, f64) {
    let signature_started_at = Instant::now();
    let mut groups_by_signature: FxHashMap<TerminalActionSignature<'a>, Vec<TerminalID>> =
        FxHashMap::default();

    for terminal in 0..grammar.num_terminals {
        groups_by_signature
            .entry(terminal_action_signature_from_index(table, index, terminal))
            .or_default()
            .push(terminal);
    }

    let mut groups: Vec<Vec<TerminalID>> = groups_by_signature.into_values().collect();
    groups.sort_by_key(|terminals| terminals.first().copied().unwrap_or(u32::MAX));
    (groups, elapsed_ms(signature_started_at))
}

fn build_characterization_index(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
) -> CharacterizationIndex {
    let mut action_states_by_terminal = vec![Vec::new(); grammar.num_terminals as usize];
    for (state, row) in table.action.iter().enumerate() {
        for (terminal, _) in row.iter() {
            if let Some(states) = action_states_by_terminal.get_mut(terminal as usize) {
                states.push(state as u32);
            }
        }
    }

    debug_assert!(
        action_states_by_terminal
            .iter()
            .all(|states| states.windows(2).all(|w| w[0] < w[1])),
        "action states should be sorted and deduplicated by construction"
    );

    let mut forwarded_only_states_by_terminal = vec![Vec::new(); grammar.num_terminals as usize];
    for &(state, terminal) in &table.forwarded_shifts {
        if state >= table.num_states || table.action(state, terminal).is_some() {
            continue;
        }
        if let Some(states) = forwarded_only_states_by_terminal.get_mut(terminal as usize) {
            states.push(state);
        }
    }
    for states in &mut forwarded_only_states_by_terminal {
        states.sort_unstable();
        states.dedup();
    }

    let mut goto_predecessors_by_target = vec![Vec::new(); table.num_states as usize];
    for (revealed_state, row) in table.goto.iter().enumerate() {
        for (&nonterminal, &(goto_state, goto_replace)) in row.iter() {
            if let Some(predecessors) = goto_predecessors_by_target.get_mut(goto_state as usize) {
                predecessors.push((revealed_state as u32, nonterminal, goto_replace));
            }
        }
    }

    CharacterizationIndex {
        action_states_by_terminal,
        forwarded_only_states_by_terminal,
        goto_predecessors_by_target,
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

fn constrain_matcher_with_sorted_allowed(matcher: &mut StackMatcher, allowed: &[u32]) -> bool {
    let next = match matcher {
        StackMatcher::Any => states_matcher(allowed.to_vec()),
        StackMatcher::State(state) => {
            if allowed.binary_search(state).is_ok() {
                Some(StackMatcher::State(*state))
            } else {
                None
            }
        }
        StackMatcher::States(states) => states_matcher(intersect_sorted(states, allowed)),
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

fn stack_effect_edits(
    config: &RelationConfig,
    shift_pop: usize,
    shifted_pushes: &[u32],
    guards: &[StackShiftGuard],
) -> Vec<StackEdit> {
    let segment_len = config.segment.len();
    let base_input_len = config.input.len();
    let mut input = config.input.clone();

    let mut previous_depth = None;
    let mut revealed_guarded = false;
    for guard in guards {
        let depth = guard.pop as usize;
        let allowed = guard.states.as_slice();

        if let Some(previous) = previous_depth {
            if depth < previous {
                return Vec::new();
            }
        }
        previous_depth = Some(depth);

        if depth > shift_pop || allowed.is_empty() {
            return Vec::new();
        }

        if depth < segment_len {
            let state = config.segment[segment_len - 1 - depth];
            if allowed.binary_search(&state).is_err() {
                return Vec::new();
            }
        } else {
            let below_segment_index = depth - segment_len;
            let input_index = base_input_len + below_segment_index;
            ensure_input_len(&mut input, input_index + 1);
            if !constrain_matcher_with_sorted_allowed(&mut input[input_index], allowed) {
                return Vec::new();
            }
        }

        if depth == shift_pop {
            revealed_guarded = true;
        }
    }

    let unknown_popped = shift_pop.saturating_sub(segment_len);
    ensure_input_len(&mut input, base_input_len + unknown_popped);

    if shift_pop < segment_len {
        let mut pushes = config.segment[..segment_len - shift_pop].to_vec();
        pushes.extend_from_slice(shifted_pushes);
        return vec![StackEdit { pop: input, pushes }];
    }

    if revealed_guarded {
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
    seen: &mut FxHashSet<RelationConfig>,
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
    seen: &mut FxHashSet<RelationConfig>,
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
    seen: &mut FxHashSet<RelationConfig>,
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
    index: &CharacterizationIndex,
    terminal: TerminalID,
    output: &mut CharacterizationOutput,
) {
    let Some(states) = index.action_states_by_terminal.get(terminal as usize) else {
        return;
    };

    for &state in states {
        let Some(action) = table.action(state, terminal) else {
            continue;
        };

        let config = identity_config(state);
        let mut seen = FxHashSet::default();
        seen.insert(config.clone());
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
    index: &CharacterizationIndex,
    terminal: TerminalID,
    output: &mut CharacterizationOutput,
) {
    let Some(action_states) = index.action_states_by_terminal.get(terminal as usize) else {
        return;
    };

    if action_states.len() < NT_CONTINUATION_PARALLEL_THRESHOLD {
        characterize_nt_continuations_for_top_states(table, index, terminal, action_states, output);
        return;
    }

    let chunk_results: Vec<CharacterizationOutput> = action_states
        .par_chunks(NT_CONTINUATION_CHUNK_SIZE)
        .map(|top_states| {
            let mut local_output = CharacterizationOutput::default();
            characterize_nt_continuations_for_top_states(
                table,
                index,
                terminal,
                top_states,
                &mut local_output,
            );
            local_output
        })
        .collect();

    for local_output in chunk_results {
        output.extend_from(local_output);
    }
}

fn characterize_nt_continuations_for_top_states(
    table: &GLRTable,
    index: &CharacterizationIndex,
    terminal: TerminalID,
    top_states: &[u32],
    output: &mut CharacterizationOutput,
) {
    for &top_state in top_states {
        let Some(predecessors) = index.goto_predecessors_by_target.get(top_state as usize) else {
            continue;
        };

        for &(revealed_state, nonterminal, goto_replace) in predecessors {
            let config = start_relation_after_goto(revealed_state, top_state, goto_replace);
            let mut seen = FxHashSet::default();
            seen.insert(config.clone());
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
    index: &CharacterizationIndex,
    terminal: TerminalID,
) -> TerminalCharacterization {
    let mut output = CharacterizationOutput::default();
    characterize_initial_actions_for_terminal(table, index, terminal, &mut output);
    characterize_nt_continuations_for_terminal(table, index, terminal, &mut output);

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
    characterize_terminals_profiled(table, grammar).0
}

fn characterize_terminals_unquotiented(
    index: &CharacterizationIndex,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
) -> BTreeMap<TerminalID, TerminalCharacterization> {
    (0..grammar.num_terminals)
        .map(|terminal| {
            let characterization = characterize_terminal(table, index, terminal);
            (terminal, characterization)
        })
        .collect()
}

fn validate_sparse_action_signatures(
    index: &CharacterizationIndex,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
) {
    for terminal in 0..grammar.num_terminals {
        let expected = dense_terminal_action_signature(table, terminal);
        let actual = owned_terminal_action_signature_from_index(table, index, terminal);
        assert_eq!(
            actual,
            expected,
            "sparse terminal action signature mismatch for terminal {terminal}"
        );
    }
}

fn validate_characterization_quotient(
    index: &CharacterizationIndex,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
) {
    for terminal in 0..grammar.num_terminals {
        let expected = characterize_terminal(table, index, terminal);
        let actual = characterizations
            .get(&terminal)
            .unwrap_or_else(|| panic!("missing characterization for terminal {terminal}"));
        assert_eq!(
            actual,
            &expected,
            "characterization quotient mismatch for terminal {terminal}"
        );
    }
}

pub(crate) fn characterize_terminals_profiled(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
) -> (BTreeMap<TerminalID, TerminalCharacterization>, TerminalCharacterizationProfile) {
    let total_started_at = Instant::now();
    let terminal_count = grammar.num_terminals as usize;
    let index = build_characterization_index(table, grammar);

    if characterization_quotient_disabled() {
        let characterize_started_at = Instant::now();
        let characterizations = characterize_terminals_unquotiented(&index, table, grammar);
        let characterize_ms = elapsed_ms(characterize_started_at);
        let total_ms = elapsed_ms(total_started_at);
        return (
            characterizations,
            TerminalCharacterizationProfile {
                terminals: terminal_count,
                unique_action_signatures: terminal_count,
                max_action_signature_multiplicity: if terminal_count == 0 { 0 } else { 1 },
                characterize_ms,
                total_ms,
                quotient_disabled: true,
                ..TerminalCharacterizationProfile::default()
            },
        );
    }

    let (groups, signature_ms) = group_terminals_by_action_signature(table, &index, grammar);
    let unique_action_signatures = groups.len();
    let max_action_signature_multiplicity = groups.iter().map(Vec::len).max().unwrap_or(0);

    let characterize_started_at = Instant::now();
    let characterized_groups: Vec<(Vec<TerminalID>, TerminalCharacterization)> = groups
        .into_par_iter()
        .map(|terminals| {
            let representative = terminals[0];
            (terminals, characterize_terminal(table, &index, representative))
        })
        .collect();
    let characterize_ms = elapsed_ms(characterize_started_at);

    let fanout_started_at = Instant::now();
    let mut characterizations = BTreeMap::new();
    for (terminals, characterization) in characterized_groups {
        for terminal in terminals {
            characterizations.insert(terminal, characterization.clone());
        }
    }
    let fanout_ms = elapsed_ms(fanout_started_at);

    let validation_started_at = Instant::now();
    if sparse_action_signature_validation_enabled() {
        validate_sparse_action_signatures(&index, table, grammar);
    }
    if characterization_quotient_validation_enabled() {
        validate_characterization_quotient(&index, table, grammar, &characterizations);
    }
    let validation_ms = elapsed_ms(validation_started_at);

    let total_ms = elapsed_ms(total_started_at);
    (
        characterizations,
        TerminalCharacterizationProfile {
            terminals: terminal_count,
            unique_action_signatures,
            max_action_signature_multiplicity,
            quotient_hits: terminal_count.saturating_sub(unique_action_signatures),
            signature_ms,
            characterize_ms,
            fanout_ms,
            validation_ms,
            total_ms,
            quotient_disabled: false,
        },
    )
}
