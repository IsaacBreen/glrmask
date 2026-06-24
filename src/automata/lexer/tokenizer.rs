//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use serde::{Deserialize, Serialize};

use super::dfa::DFA;
use crate::automata::regex::Expr;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub(super) dfa: DFA,
    pub(super) num_terminals: u32,
    /// Per-terminal regex expressions used to (re)build this tokenizer.
    /// Skipped during (de)serialization because they are only needed during
    /// compile-time simplification for active-terminal rebuilds.
    #[serde(default, skip)]
    pub(super) exprs: Option<Arc<[Expr]>>,
    #[serde(skip, default = "new_group_dfa_cache")]
    group_dfas: Arc<OnceLock<Arc<[GroupDfa]>>>,
}

#[derive(Debug, Clone)]
pub(crate) struct GroupDfa {
    pub(crate) dfa: DFA,
    pub(crate) joint_state_to_group_state: Arc<[u32]>,
}

impl GroupDfa {
    pub(crate) fn num_states(&self) -> usize {
        self.dfa.num_states()
    }

    pub(crate) fn is_match(&self, state: u32) -> bool {
        self.dfa.finalizers(state).contains(0)
    }

    pub(crate) fn can_continue(&self, state: u32) -> bool {
        self.dfa.possible_future_group_ids(state).contains(0)
    }

    pub(crate) fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.dfa.get_transition(state, byte)
    }
}

fn new_group_dfa_cache() -> Arc<OnceLock<Arc<[GroupDfa]>>> {
    Arc::new(OnceLock::new())
}

fn build_group_dfas(joint_dfa: &DFA, num_terminals: u32) -> Arc<[GroupDfa]> {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
    let started_at = profile_enabled.then(Instant::now);
    let group_dfas = (0..num_terminals)
        .map(|terminal| build_group_dfa(joint_dfa, terminal))
        .collect::<Vec<_>>()
        ;
    if let Some(started_at) = started_at {
        let states: usize = group_dfas.iter().map(|group| group.dfa.num_states()).sum();
        let transitions: usize = group_dfas
            .iter()
            .flat_map(|group| group.dfa.states())
            .map(|state| state.transitions.len())
            .sum();
        eprintln!(
            "[glrmask/profile][tokenizer_group_dfas] groups={} joint_states={} isolated_states={} isolated_transitions={} total_ms={:.3}",
            num_terminals,
            joint_dfa.num_states(),
            states,
            transitions,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    group_dfas.into()
}

fn build_group_dfa(joint_dfa: &DFA, terminal: TerminalID) -> GroupDfa {
    let joint_state_count = joint_dfa.num_states();
    let terminal_index = terminal as usize;
    let mut joint_state_to_group_state = vec![u32::MAX; joint_state_count];
    let mut dfa = DFA::new(1);
    dfa.ensure_group_capacity(1);
    dfa.set_group_u8set(0, *joint_dfa.group_id_to_u8set(terminal));

    let joint_starts = joint_dfa.start_states().to_vec();
    let joint_start = joint_dfa.start_state();
    joint_state_to_group_state[joint_start as usize] = 0;
    let mut queue = VecDeque::from([joint_start]);
    let mut group_starts = vec![0];
    for joint_start in joint_starts.into_iter().skip(1) {
        let local_start = if joint_state_to_group_state[joint_start as usize] == u32::MAX {
            let local_start = dfa.add_state();
            joint_state_to_group_state[joint_start as usize] = local_start;
            queue.push_back(joint_start);
            local_start
        } else {
            joint_state_to_group_state[joint_start as usize]
        };
        group_starts.push(local_start);
    }

    while let Some(joint_state) = queue.pop_front() {
        let local_state = joint_state_to_group_state[joint_state as usize];
        let joint = &joint_dfa.states()[joint_state as usize];
        let is_match = joint.finalizers.contains(terminal_index);
        let is_live = is_match
            || joint_dfa
                .possible_future_group_ids(joint_state)
                .contains(terminal_index);

        if !is_live && joint_state != joint_start {
            continue;
        }

        let mut finalizers = BitSet::new(1);
        if is_match {
            finalizers.set(0);
        }
        dfa.overwrite_state_metadata(local_state, finalizers, BitSet::new(1));

        let mut transitions = Vec::new();
        for (byte, &joint_target) in joint.transitions.iter() {
            let target = &joint_dfa.states()[joint_target as usize];
            let target_live = target.finalizers.contains(terminal_index)
                || joint_dfa
                    .possible_future_group_ids(joint_target)
                    .contains(terminal_index);
            if !target_live {
                continue;
            }

            let local_target = if joint_state_to_group_state[joint_target as usize] == u32::MAX {
                let local_target = dfa.add_state();
                joint_state_to_group_state[joint_target as usize] = local_target;
                queue.push_back(joint_target);
                local_target
            } else {
                joint_state_to_group_state[joint_target as usize]
            };
            transitions.push((byte, local_target));
        }
        dfa.set_transitions_from_sorted_entries(local_state, transitions);
    }

    dfa.set_start_states(group_starts);

    let (minimized, local_state_map) = dfa.minimize_with_state_mapping();
    for state in &mut joint_state_to_group_state {
        if *state != u32::MAX {
            *state = local_state_map[*state as usize];
        }
    }

    GroupDfa {
        dfa: minimized,
        joint_state_to_group_state: Arc::from(joint_state_to_group_state),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMatch {
    pub id: TerminalID,
    pub width: usize,
    pub end_state: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExecResult {
    pub end_state: Option<u32>,
    pub matches: Vec<TokenizerMatch>,
}

pub(crate) trait Lexer {
    fn start_state(&self) -> u32;
    fn num_terminals(&self) -> u32;
    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_;

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        for (byte, target) in self.transitions_from(state) {
            row[byte as usize] = target;
        }
    }

    fn transition_row(&self, state: u32) -> Box<[u32; 256]> {
        let mut row = Box::new([u32::MAX; 256]);
        self.fill_transition_row(state, &mut row);
        row
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        for (byte, target) in self.transitions_from(state) {
            if target == state {
                bytes.insert(byte);
            }
        }
        bytes
    }

    fn transition_count(&self) -> usize {
        (0..self.num_states())
            .map(|state| self.transitions_from(state).count())
            .sum()
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32>;
    fn get_transition(&self, state: u32, byte: u8) -> u32;
    fn matched_terminal_bitset(&self, state: u32) -> &BitSet;
    fn matched_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_;
    fn possible_future_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_;
    fn possible_future_terminals(&self, state: u32) -> &BitSet;

    fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    fn num_states(&self) -> u32;
    fn num_forced_minimized_states(&self) -> usize;
    fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult;
    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult;
    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> Option<u32>;
    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult;

    fn initial_state(&self) -> u32 {
        self.start_state()
    }

    fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, Option<u32>);
}

fn into_longest_matches(matches: FxHashMap<TerminalID, (usize, u32)>) -> Vec<TokenizerMatch> {
    matches
        .into_iter()
        .map(|(id, (width, end_state))| TokenizerMatch {
            id,
            width,
            end_state,
        })
        .collect()
}

fn group_matches_by_width(matches: Vec<TokenizerMatch>) -> Vec<(usize, BTreeSet<TerminalID>)> {
    let mut grouped = std::collections::BTreeMap::<usize, BTreeSet<TerminalID>>::new();
    for matched in matches {
        grouped.entry(matched.width).or_default().insert(matched.id);
    }
    grouped.into_iter().collect()
}

struct TerminalFilteredDfa {
    dfa: DFA,
    active_bitset: BitSet,
    any_cleared: bool,
    transitions_pruned: bool,
}

impl Tokenizer {
    pub(super) fn from_parts(
        dfa: DFA,
        num_terminals: u32,
        exprs: Option<Arc<[Expr]>>,
    ) -> Self {
        Self {
            dfa,
            num_terminals,
            exprs,
            group_dfas: new_group_dfa_cache(),
        }
    }

    fn invalidate_group_dfas(&mut self) {
        self.group_dfas = new_group_dfa_cache();
    }

    fn group_dfas(&self) -> &[GroupDfa] {
        self.group_dfas
            .get_or_init(|| build_group_dfas(&self.dfa, self.num_terminals))
            .as_ref()
    }

    /// Materialize and retain every isolated group DFA. The terminal-
    /// equivalence builder invokes this through `group_dfa`; other callers can
    /// request it explicitly.
    pub fn materialize_group_dfas(&self) {
        let _ = self.group_dfas();
    }

    /// Number of isolated group DFAs retained by this tokenizer.
    pub fn group_dfa_count(&self) -> usize {
        self.group_dfas().len()
    }

    pub(crate) fn group_dfa(&self, terminal: TerminalID) -> &GroupDfa {
        &self.group_dfas()[terminal as usize]
    }

    /// The default lexer entry state.
    pub fn start_state(&self) -> u32 {
        self.dfa.start_state()
    }

    /// All selectable lexer entry states, with [`Self::start_state`] first.
    pub fn start_states(&self) -> &[u32] {
        self.dfa.start_states()
    }

    /// Replace all selectable lexer entry states. The first state is the
    /// default used by `run()` and legacy callers.
    pub fn set_start_states(&mut self, states: Vec<u32>) {
        self.dfa.set_start_states(states);
        self.invalidate_group_dfas();
    }

    /// Select the default lexer entry state while retaining all other entries.
    pub fn set_default_start_state(&mut self, state: u32) {
        self.dfa.set_default_start_state(state);
        self.invalidate_group_dfas();
    }

    /// Add an auxiliary selectable lexer entry state.
    pub fn add_start_state(&mut self, state: u32) {
        self.dfa.add_start_state(state);
        self.invalidate_group_dfas();
    }

    /// Number of states in the isolated DFA for `terminal`.
    pub fn group_dfa_num_states(&self, terminal: TerminalID) -> usize {
        self.group_dfa(terminal).dfa.num_states()
    }

    /// Default state of the isolated DFA for `terminal`.
    pub fn group_dfa_start_state(&self, terminal: TerminalID) -> u32 {
        self.group_dfa(terminal).dfa.start_state()
    }

    /// All selectable entries of `terminal`'s isolated DFA, with its default
    /// first. This mirrors generic multi-entry lexer semantics when present.
    pub fn group_dfa_start_states(&self, terminal: TerminalID) -> Vec<u32> {
        self.group_dfa(terminal).dfa.start_states().to_vec()
    }

    /// Step the isolated DFA for `terminal`. Its group-0 match predicate means
    /// this original terminal and no other terminal.
    pub fn group_dfa_step(&self, terminal: TerminalID, state: u32, byte: u8) -> Option<u32> {
        self.group_dfa(terminal).dfa.step(state, byte)
    }

    /// Whether `state` in `terminal`'s isolated DFA accepts that terminal.
    pub fn group_dfa_is_match(&self, terminal: TerminalID, state: u32) -> bool {
        self.group_dfa(terminal).dfa.finalizers(state).contains(0)
    }

    /// Whether `state` in `terminal`'s isolated DFA has a non-empty future.
    pub fn group_dfa_can_continue(&self, terminal: TerminalID, state: u32) -> bool {
        !self
            .group_dfa(terminal)
            .dfa
            .possible_future_group_ids(state)
            .is_empty()
    }

    fn num_terminals(&self) -> u32 {
        self.num_terminals
    }

    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.dfa
            .states()
            .get(state as usize)
            .into_iter()
            .flat_map(|state| state.transitions.iter().map(|(byte, &target)| (byte, target)))
    }

    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) {
        row.fill(u32::MAX);
        for (byte, target) in self.transitions_from(state) {
            row[byte as usize] = target;
        }
    }

    fn transition_row(&self, state: u32) -> Box<[u32; 256]> {
        let mut row = Box::new([u32::MAX; 256]);
        self.fill_transition_row(state, &mut row);
        row
    }

    fn self_loop_bytes(&self, state: u32) -> U8Set {
        let mut bytes = U8Set::empty();
        for (byte, target) in self.transitions_from(state) {
            if target == state {
                bytes.insert(byte);
            }
        }
        bytes
    }

    fn transition_count(&self) -> usize {
        (0..self.num_states())
            .map(|state| self.transitions_from(state).count())
            .sum()
    }

    /// Detect terminals nullable from any selectable entry state, remove those
    /// entry-state finalizers from the DFA, and return their union. After this
    /// call the tokenizer reports no zero-byte terminal match from any entry.
    pub fn isolate_start_state_and_drain_nullable_terminals(&mut self) -> BTreeSet<TerminalID> {
        self.isolate_start_state();
        let mut nullable = BTreeSet::new();
        for start in self.start_states().to_vec() {
            nullable.extend(
                self.dfa
                    .clear_finalizers_for_state(start)
                    .iter()
                    .map(|terminal| terminal as TerminalID),
            );
        }
        self.invalidate_group_dfas();
        nullable
    }

    /// Ensure that no byte transition in the DFA targets a selectable entry
    /// state.
    ///
    /// If any transition does, a copy of the start state is created and all
    /// such transitions are redirected to the copy. This keeps the DFA
    /// equivalent while guaranteeing every designated entry state is only
    /// reached at position 0 when that entry was explicitly selected.
    fn isolate_start_state(&mut self) {
        let starts = self.dfa.start_states().to_vec();
        for start in starts {
            let clone_id = self.dfa.clone_state(start);
            if !self.dfa.redirect_transitions(start, clone_id) {
                self.dfa.discard_last_state(clone_id);
            }
        }
        self.invalidate_group_dfas();
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.dfa.get_transition(state, byte)
    }

    pub fn run(&self, input: &[u8]) -> u32 {
        input
            .iter()
            .try_fold(self.start_state(), |state, &byte| self.step(state, byte))
            .unwrap_or(self.start_state())
    }

    /// Run from a selected lexer entry or continuation state. Unlike `run`,
    /// failure remains observable rather than falling back to the default
    /// start state.
    pub fn run_from_state(&self, input: &[u8], start: u32) -> Option<u32> {
        input
            .iter()
            .try_fold(start, |state, &byte| self.step(state, byte))
    }

    pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.matched_terminals_iter(state).collect()
    }

    fn matched_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.dfa
            .finalizers(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    fn matched_terminal_bitset(&self, state: u32) -> &BitSet {
        self.dfa.finalizers(state)
    }

    fn possible_future_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.dfa
            .possible_future_group_ids(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    fn possible_future_terminals(&self, state: u32) -> &BitSet {
        self.dfa.possible_future_group_ids(state)
    }

    fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    fn num_states(&self) -> u32 {
        self.dfa.num_states() as u32
    }

    fn num_forced_minimized_states(&self) -> usize {
        self.dfa.minimize().num_states()
    }

    fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult {
        let mut matches = Vec::new();
        let end_state = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_all_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state: end_state.filter(|&state| !self.is_end(state)),
            matches,
        }
    }

    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {
        let mut matches = FxHashMap::<TerminalID, (usize, u32)>::default();
        let end_state = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_longest_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state,
            matches: into_longest_matches(matches),
        }
    }

    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> Option<u32> {
        self.scan_input(input, start, &mut (), |_, _, _, _| {})
    }

    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
        let exec = self.execute_from_state_all_widths(input, start);
        let end_state = exec.end_state.unwrap_or(start);
        TokenizerResult {
            end_state,
            matches: group_matches_by_width(exec.matches),
        }
    }

    fn initial_state(&self) -> u32 {
        self.start_state()
    }

    fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    /// Scan input bytes and report which terminals of interest matched/finalized.
    ///
    /// Returns a bitset of matched terminals and an optional end state.
    ///
    /// Algorithm:
    /// 1. `remaining = terminals_of_interest`.
    /// 2. `matched = empty`.
    /// 3. For each byte:
    ///    - Check if current state's possible futures overlap `remaining`.
    ///      If not, return `(matched, None)`.
    ///    - Consume byte → next state.
    ///    - If no transition, return `(matched, None)`.
    ///    - Get finalizers at next state, intersect with `remaining`.
    ///    - Add intersection to `matched`, remove from `remaining`.
    /// 4. After all bytes, check futures at end state overlap `remaining`.
    ///    If not, return `(matched, None)`. Otherwise `(matched, Some(end_state))`.
    ///
    /// Important: initial-state finalizers are intentionally ignored.
    /// Only post-byte finalizers count.
    ///
    /// `terminals_of_interest` must have length equal to `self.num_terminals`.
    fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, Option<u32>) {
        debug_assert_eq!(terminals_of_interest.len(), self.num_terminals as usize);
        let mut remaining = terminals_of_interest.clone();
        let mut matched = BitSet::new(self.num_terminals as usize);
        let mut state = start;

        for &byte in input {
            let futures = self.possible_future_terminals(state);
            if futures.is_disjoint(&remaining) {
                return (matched, None);
            }

            let next = match self.step(state, byte) {
                Some(s) => s,
                None => return (matched, None),
            };

            let finals = self.dfa.finalizers(next).intersection(&remaining);
            matched.union_with(&finals);
            remaining = remaining.difference(&finals);
            state = next;
        }

        let futures = self.possible_future_terminals(state);
        if futures.is_disjoint(&remaining) {
            (matched, None)
        } else {
            (matched, Some(state))
        }
    }

    fn record_all_matches(&self, matches: &mut Vec<TokenizerMatch>, state: u32, width: usize) {
        matches.extend(self.matched_terminals_iter(state).map(|id| TokenizerMatch {
            id,
            width,
            end_state: state,
        }));
    }

    fn record_longest_matches(
        &self,
        matches: &mut FxHashMap<TerminalID, (usize, u32)>,
        state: u32,
        width: usize,
    ) {
        for terminal in self.matched_terminals_iter(state) {
            matches.insert(terminal, (width, state));
        }
    }

    fn scan_input<R>(
        &self,
        input: &[u8],
        start: u32,
        mut matches: &mut R,
        mut record_matches: impl FnMut(&Self, &mut R, u32, usize),
    ) -> Option<u32> {
        let mut state = start;
        for (index, &byte) in input.iter().enumerate() {
            let next = self.step(state, byte)?;
            state = next;
            record_matches(self, &mut matches, state, index + 1);
        }
        Some(state)
    }

    fn filter_dfa_for_terminals(
        &self,
        active_terminals: &[bool],
        relevant_bytes: Option<&[bool; 256]>,
    ) -> TerminalFilteredDfa {
        let mut dfa = self.dfa.clone();

        let num_groups = self.num_terminals as usize;
        let mut active_bitset = BitSet::new(num_groups);
        for (tid, &active) in active_terminals.iter().enumerate() {
            if active {
                active_bitset.set(tid);
            }
        }

        let mut any_cleared = false;
        let mut transitions_pruned = false;
        for state in dfa.states_mut().iter_mut() {
            if let Some(relevant_bytes) = relevant_bytes {
                let mut filtered_transitions = Vec::with_capacity(state.transitions.len());
                for (byte, &target) in state.transitions.iter() {
                    if relevant_bytes[byte as usize] {
                        filtered_transitions.push((byte, target));
                    }
                }
                if filtered_transitions.len() != state.transitions.len() {
                    state.transitions =
                        crate::ds::char_transitions::CharTransitions::from_sorted_entries(
                            filtered_transitions,
                        );
                    transitions_pruned = true;
                }
            }

            if state.finalizers.len() == active_bitset.len()
                && !state.finalizers.is_subset(&active_bitset)
            {
                state.finalizers.intersect_with(&active_bitset);
                any_cleared = true;
            } else {
                for (terminal_id, active) in active_terminals.iter().enumerate() {
                    if !active
                        && terminal_id < state.finalizers.len()
                        && state.finalizers.contains(terminal_id)
                    {
                        state.finalizers.clear(terminal_id);
                        any_cleared = true;
                    }
                }
            }
        }

        let num_states_after_filter = dfa.num_states();
        let mut is_dead = vec![false; num_states_after_filter];
        for state_id in 0..num_states_after_filter {
            let state = &dfa.states()[state_id];
            let final_active = !state.finalizers.is_disjoint(&active_bitset);
            let future_active = !self
                .dfa
                .possible_future_group_ids(state_id as u32)
                .is_disjoint(&active_bitset);
            if !final_active && !future_active {
                is_dead[state_id] = true;
            }
        }
        let mut coreach_pruned = false;
        for state in dfa.states_mut().iter_mut() {
            let original_len = state.transitions.len();
            if original_len == 0 {
                continue;
            }
            let mut filtered = Vec::with_capacity(original_len);
            for (byte, &target) in state.transitions.iter() {
                if !is_dead[target as usize] {
                    filtered.push((byte, target));
                }
            }
            if filtered.len() != original_len {
                state.transitions =
                    crate::ds::char_transitions::CharTransitions::from_sorted_entries(filtered);
                coreach_pruned = true;
            }
        }
        if coreach_pruned {
            any_cleared = true;
        }

        TerminalFilteredDfa {
            dfa,
            active_bitset,
            any_cleared,
            transitions_pruned,
        }
    }

    /// Create a simplified tokenizer that only knows about `active_terminals`.
    ///
    /// Non-active terminal bits are cleared from finalizers and the filtered
    /// DFA is then minimized when that preserves the required mapping shape.
    pub fn simplify_for_terminals(
        &self,
        active_terminals: &[bool],
        relevant_bytes: Option<&[bool; 256]>,
    ) -> (Tokenizer, ManyToOneIdMap) {
        let TerminalFilteredDfa {
            mut dfa,
            active_bitset,
            any_cleared,
            transitions_pruned,
        } = self.filter_dfa_for_terminals(active_terminals, relevant_bytes);

        if !any_cleared && !transitions_pruned {
            let num_states = dfa.num_states();
            let identity = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                (0..num_states as u32).collect(),
                num_states as u32,
            );
            return (
                Tokenizer::from_parts(dfa, self.num_terminals, self.exprs.clone()),
                identity,
            );
        }

        let pre_minimize_states = dfa.num_states();
        let num_active = active_terminals.iter().filter(|&&active| active).count();
        if pre_minimize_states > 1000 && num_active > 32 && !transitions_pruned {
            let distinct = dfa.distinct_fingerprint_count();
            if distinct > pre_minimize_states * 9 / 10 {
                dfa.mask_possible_futures(&active_bitset);
                let identity = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                    (0..pre_minimize_states as u32).collect(),
                    pre_minimize_states as u32,
                );
                return (
                    Tokenizer::from_parts(dfa, self.num_terminals, self.exprs.clone()),
                    identity,
                );
            }
        }

        if transitions_pruned {
            // Downstream L1/L2P composition treats the returned map as an
            // original-state map. Minimizing a byte-pruned DFA can merge
            // continuation states that must stay distinct for whole-token
            // signatures, so keep the filtered DFA and an identity map.
            dfa.mask_possible_futures(&active_bitset);
            let num_states = dfa.num_states();
            let identity = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                (0..num_states as u32).collect(),
                num_states as u32,
            );
            return (
                Tokenizer::from_parts(dfa, self.num_terminals, self.exprs.clone()),
                identity,
            );
        }

        let (minimized, state_mapping) = dfa.minimize_with_state_mapping();
        let post_minimize_states = minimized.num_states();

        (
            Tokenizer::from_parts(minimized, self.num_terminals, self.exprs.clone()),
            ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                state_mapping,
                post_minimize_states as u32,
            ),
        )
    }

    /// Exact active-language DFA quotient.
    ///
    /// Inactive finalizers and transitions into states that cannot complete an
    /// active terminal are removed before minimization. The resulting quotient
    /// is equivalent to rebuilding the lexer from only the active terminals.
    /// Original states outside that language are intentionally left unmapped.
    pub(crate) fn minimize_for_active_finalizers(
        &self,
        active_groups: Option<&[bool]>,
    ) -> ManyToOneIdMap {
        if let Some(active_groups) = active_groups {
            let TerminalFilteredDfa { dfa, .. } =
                self.filter_dfa_for_terminals(active_groups, None);
            let (minimized, state_mapping) = dfa.minimize_with_state_mapping();
            return ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                state_mapping,
                minimized.num_states() as u32,
            );
        }

        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let started_at = profile_enabled.then(std::time::Instant::now);
        let num_states = self.num_states() as usize;
        if num_states == 0 {
            return ManyToOneIdMap::from_original_to_internal_allowing_unmapped(Vec::new(), 0);
        }

        let active_mask = active_groups.map(|active_groups| {
            let mut active = BitSet::new(self.dfa.num_groups());
            for (group, &is_active) in active_groups.iter().enumerate() {
                if is_active && group < self.dfa.num_groups() {
                    active.set(group);
                }
            }
            active
        });

        let mut finalizer_to_block = FxHashMap::<BitSet, u32>::default();
        let mut blocks = vec![0u32; num_states];
        let mut block_count = 0u32;
        for (state_id, state) in self.dfa.states().iter().enumerate() {
            let mut finalizers = state.finalizers.clone();
            if let Some(active_mask) = &active_mask {
                finalizers.intersect_with(active_mask);
            }
            let block = *finalizer_to_block.entry(finalizers).or_insert_with(|| {
                let block = block_count;
                block_count += 1;
                block
            });
            blocks[state_id] = block;
        }

        let mut iterations = 0usize;
        loop {
            let mut next_blocks = vec![0u32; num_states];
            let mut buckets = FxHashMap::<u64, SmallVec<[(usize, u32); 1]>>::default();
            let mut next_block_count = 0u32;

            for state_id in 0..num_states {
                let transitions = &self.dfa.states()[state_id].transitions;
                let mut hasher = FxHasher::default();
                blocks[state_id].hash(&mut hasher);
                transitions.len().hash(&mut hasher);
                for (byte, &target) in transitions.iter() {
                    byte.hash(&mut hasher);
                    blocks[target as usize].hash(&mut hasher);
                }
                let candidates = buckets.entry(hasher.finish()).or_default();
                let block = candidates
                    .iter()
                    .find_map(|&(representative, block)| {
                        active_finalizer_refinement_rows_equal(
                            &self.dfa,
                            state_id,
                            representative,
                            &blocks,
                        )
                        .then_some(block)
                    })
                    .unwrap_or_else(|| {
                        let block = next_block_count;
                        next_block_count += 1;
                        candidates.push((state_id, block));
                        block
                    });
                next_blocks[state_id] = block;
            }

            iterations += 1;
            if next_block_count == block_count {
                blocks = next_blocks;
                break;
            }
            blocks = next_blocks;
            block_count = next_block_count;
        }

        if let Some(started_at) = started_at {
            eprintln!(
                "[glrmask/profile][active_finalizer_dfa_minimize] states={} active_groups={} initial_blocks={} final_blocks={} iterations={} total_ms={:.3}",
                num_states,
                active_groups.map_or(self.dfa.num_groups(), |groups| groups.iter().filter(|&&active| active).count()),
                finalizer_to_block.len(),
                block_count,
                iterations,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        ManyToOneIdMap::from_original_to_internal_allowing_unmapped(blocks, block_count)
    }

    /// Rebuild the lexer from only the active terminal expressions, then
    /// minimize that rebuilt lexer. This is a validation oracle for
    /// `minimize_for_active_finalizers(Some(active_groups))`: both describe
    /// the same labelled active-terminal language.
    pub(crate) fn rebuilt_active_terminal_minimized_state_count(
        &self,
        active_groups: &[bool],
    ) -> Option<usize> {
        let expressions = self.exprs.as_ref()?;
        assert_eq!(
            expressions.len(),
            self.num_terminals as usize,
            "terminal-expression inventory must match tokenizer groups",
        );
        let active_expressions: Vec<Expr> = expressions
            .iter()
            .zip(active_groups.iter().copied())
            .filter_map(|(expression, active)| active.then(|| expression.clone()))
            .collect();
        let rebuilt = super::compile::build_regex(&active_expressions).into_tokenizer(
            active_expressions.len() as u32,
            Some(Arc::from(active_expressions.into_boxed_slice())),
        );
        Some(
            rebuilt
                .minimize_for_active_finalizers(None)
                .num_internal_ids() as usize,
        )
    }

    /// Relabel terminal finalizer/future observations without changing the
    /// state graph. Every concrete member remains live, but is observed as its
    /// terminal-class representative by the class-level state analysis.
    pub(crate) fn relabel_for_terminal_labels(
        &self,
        terminal_labels: &[TerminalID],
        active_labels: &[bool],
    ) -> Tokenizer {
        Tokenizer::from_parts(
            self.relabel_dfa_for_terminal_labels(terminal_labels, active_labels),
            self.num_terminals,
            self.exprs.clone(),
        )
    }

    fn relabel_dfa_for_terminal_labels(
        &self,
        terminal_labels: &[TerminalID],
        active_labels: &[bool],
    ) -> DFA {
        assert_eq!(
            terminal_labels.len(),
            self.num_terminals as usize,
            "terminal label map must cover every tokenizer terminal",
        );

        let mut dfa = self.dfa.clone();
        let num_labels = self.num_terminals as usize;
        dfa.ensure_group_capacity(num_labels);

        for state in 0..self.num_states() {
            let mut finalizers = BitSet::new(num_labels);
            for terminal in self.matched_terminals_iter(state) {
                let label = terminal_labels[terminal as usize] as usize;
                if active_labels.get(label).copied().unwrap_or(false) {
                    finalizers.set(label);
                }
            }

            let mut futures = BitSet::new(num_labels);
            for terminal in self.possible_future_terminals_iter(state) {
                let label = terminal_labels[terminal as usize] as usize;
                if active_labels.get(label).copied().unwrap_or(false) {
                    futures.set(label);
                }
            }

            dfa.overwrite_state_metadata(state, finalizers, futures);
        }

        dfa
    }

}


fn active_finalizer_refinement_rows_equal(
    dfa: &DFA,
    left: usize,
    right: usize,
    blocks: &[u32],
) -> bool {
    if blocks[left] != blocks[right] {
        return false;
    }
    let left_transitions = &dfa.states()[left].transitions;
    let right_transitions = &dfa.states()[right].transitions;
    if left_transitions.len() != right_transitions.len() {
        return false;
    }
    left_transitions
        .iter()
        .zip(right_transitions.iter())
        .all(|((left_byte, &left_target), (right_byte, &right_target))| {
            left_byte == right_byte && blocks[left_target as usize] == blocks[right_target as usize]
        })
}

impl Lexer for Tokenizer {
    fn start_state(&self) -> u32 { self.start_state() }
    fn num_terminals(&self) -> u32 { self.num_terminals() }
    fn transitions_from(&self, state: u32) -> impl Iterator<Item = (u8, u32)> + '_ { self.transitions_from(state) }
    fn fill_transition_row(&self, state: u32, row: &mut [u32; 256]) { self.fill_transition_row(state, row); }
    fn transition_row(&self, state: u32) -> Box<[u32; 256]> { self.transition_row(state) }
    fn self_loop_bytes(&self, state: u32) -> U8Set { self.self_loop_bytes(state) }
    fn transition_count(&self) -> usize { self.transition_count() }
    fn step(&self, state: u32, byte: u8) -> Option<u32> { self.step(state, byte) }
    fn get_transition(&self, state: u32, byte: u8) -> u32 { self.get_transition(state, byte) }
    fn matched_terminal_bitset(&self, state: u32) -> &BitSet { self.matched_terminal_bitset(state) }
    fn matched_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_ { self.matched_terminals_iter(state) }
    fn possible_future_terminals_iter(&self, state: u32) -> impl Iterator<Item = TerminalID> + '_ { self.possible_future_terminals_iter(state) }
    fn possible_future_terminals(&self, state: u32) -> &BitSet { self.possible_future_terminals(state) }
    fn is_end(&self, state: u32) -> bool { self.is_end(state) }
    fn num_states(&self) -> u32 { self.num_states() }
    fn num_forced_minimized_states(&self) -> usize { self.num_forced_minimized_states() }
    fn execute_from_state_all_widths(&self, input: &[u8], start: u32) -> TokenizerExecResult { self.execute_from_state_all_widths(input, start) }
    fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult { self.execute_from_state(input, start) }
    fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> Option<u32> { self.execute_from_state_end_only(input, start) }
    fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult { self.execute_all_matches(input, start) }
    fn initial_state(&self) -> u32 { self.initial_state() }
    fn initial_state_id(&self) -> u32 { self.initial_state_id() }
    fn tokens_accessible_from_state(&self, state: u32) -> &BitSet { self.tokens_accessible_from_state(state) }
    fn scan_terminal_matches_from_state(&self, input: &[u8], start: u32, terminals_of_interest: &BitSet) -> (BitSet, Option<u32>) {
        self.scan_terminal_matches_from_state(input, start, terminals_of_interest)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    pub end_state: u32,
    pub matches: Vec<(usize, BTreeSet<TerminalID>)>,
}

#[cfg(test)]
mod active_finalizer_minimization_tests {
    use super::*;
    use crate::automata::lexer::compile::build_regex;
    use crate::automata::lexer::dfa::DFA;

    #[derive(Serialize)]
    struct LegacyTokenizerWire {
        dfa: DFA,
        num_terminals: u32,
    }

    #[test]
    fn active_finalizer_refinement_matches_rebuilt_active_lexer() {
        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"ba".to_vec()),
            Expr::U8Seq(b"bb".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let active = [true, false, true, false];
        let actual = tokenizer.minimize_for_active_finalizers(Some(&active));
        let rebuilt_count = tokenizer
            .rebuilt_active_terminal_minimized_state_count(&active)
            .expect("test tokenizer keeps terminal expressions");

        assert_eq!(actual.num_internal_ids() as usize, rebuilt_count);
    }

    #[test]
    fn tokenizer_exposes_and_isolates_all_selectable_entry_states() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, b'a', 2);
        dfa.add_transition(1, b'b', 2);
        dfa.add_transition(2, b'a', 0);
        dfa.add_transition(2, b'b', 1);
        dfa.set_start_states(vec![0, 1]);

        let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);
        assert_eq!(tokenizer.start_states(), &[0, 1]);
        assert_eq!(tokenizer.run(b"a"), 2);
        assert_eq!(tokenizer.run_from_state(b"b", 1), Some(2));
        assert_eq!(tokenizer.run_from_state(b"b", 0), None);

        tokenizer.set_default_start_state(1);
        assert_eq!(tokenizer.start_states(), &[1, 0]);
        assert_eq!(tokenizer.run(b"b"), 2);
        tokenizer.set_default_start_state(0);

        tokenizer.isolate_start_state();
        let entry_states = tokenizer.start_states().to_vec();
        for source in 0..tokenizer.num_states() {
            for (_, target) in tokenizer.transitions_from(source) {
                assert!(
                    !entry_states.contains(&target),
                    "entry state {target} remained reachable from state {source}",
                );
            }
        }
        assert_eq!(tokenizer.run(b"a"), 2);
        assert_eq!(tokenizer.run_from_state(b"b", 1), Some(2));
    }

    #[test]
    fn tokenizer_serialization_preserves_selectable_entry_states() {
        let mut dfa = DFA::new(3);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, b'a', 2);
        dfa.add_transition(1, b'b', 2);
        dfa.set_start_states(vec![0, 1]);
        let tokenizer = Tokenizer::from_parts(dfa, 1, None);

        let bytes = bincode::serialize(&tokenizer).expect("tokenizer serializes");
        let restored: Tokenizer = bincode::deserialize(&bytes).expect("tokenizer deserializes");
        assert_eq!(restored.start_states(), &[0, 1]);
        assert_eq!(restored.run_from_state(b"b", 1), Some(2));
    }

    #[test]
    fn nullable_drain_clears_every_selectable_entry_state() {
        let mut dfa = DFA::new(2);
        dfa.ensure_group_capacity(1);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(0, finalizers.clone(), BitSet::new(1));
        dfa.overwrite_state_metadata(1, finalizers, BitSet::new(1));
        dfa.set_start_states(vec![0, 1]);
        let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);

        assert_eq!(
            tokenizer.isolate_start_state_and_drain_nullable_terminals(),
            BTreeSet::from([0]),
        );
        for &start in tokenizer.start_states() {
            assert!(tokenizer.matched_terminal_bitset(start).is_empty());
        }
    }

    #[test]
    fn group_dfas_are_isolated_and_map_joint_residuals() {
        let expressions = vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"x".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        let joint_start = tokenizer.start_state();
        let joint_after_a = tokenizer
            .run_from_state(b"a", joint_start)
            .expect("joint lexer accepts shared prefix");
        let joint_after_x = tokenizer
            .run_from_state(b"x", joint_start)
            .expect("joint lexer accepts x");

        for terminal in 0..3u32 {
            let group = tokenizer.group_dfa(terminal);
            assert_eq!(group.dfa.num_groups(), 1);
            for (state_id, state) in group.dfa.states().iter().enumerate() {
                assert!(state.finalizers.count_ones() <= 1);
                assert!(
                    group
                        .dfa
                        .possible_future_group_ids(state_id as u32)
                        .count_ones()
                        <= 1
                );
            }
            assert_ne!(
                group.joint_state_to_group_state[joint_start as usize],
                u32::MAX,
            );
        }

        let ab_start = tokenizer.group_dfa_start_state(0);
        let ab_after_a = tokenizer
            .group_dfa_step(0, ab_start, b'a')
            .expect("ab group accepts shared prefix");
        let ab_end = tokenizer
            .group_dfa_step(0, ab_after_a, b'b')
            .expect("ab group accepts ab");
        assert!(tokenizer.group_dfa_is_match(0, ab_end));
        assert_eq!(tokenizer.group_dfa_step(0, ab_after_a, b'c'), None);

        let ac_start = tokenizer.group_dfa_start_state(1);
        let ac_after_a = tokenizer
            .group_dfa_step(1, ac_start, b'a')
            .expect("ac group accepts shared prefix");
        let ac_end = tokenizer
            .group_dfa_step(1, ac_after_a, b'c')
            .expect("ac group accepts ac");
        assert!(tokenizer.group_dfa_is_match(1, ac_end));
        assert_eq!(tokenizer.group_dfa_step(1, ac_after_a, b'b'), None);

        assert_ne!(
            tokenizer.group_dfa(0).joint_state_to_group_state[joint_after_a as usize],
            u32::MAX,
        );
        assert_ne!(
            tokenizer.group_dfa(1).joint_state_to_group_state[joint_after_a as usize],
            u32::MAX,
        );
        assert_eq!(
            tokenizer.group_dfa(0).joint_state_to_group_state[joint_after_x as usize],
            u32::MAX,
        );
        assert_ne!(
            tokenizer.group_dfa(2).joint_state_to_group_state[joint_after_x as usize],
            u32::MAX,
        );
    }

    #[test]
    fn group_dfas_rebuild_after_tokenizer_round_trip() {
        let expressions = vec![Expr::U8Seq(b"ab".to_vec()), Expr::U8Seq(b"x".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        assert_eq!(tokenizer.group_dfa_num_states(0), 3);

        let bytes = bincode::serialize(&tokenizer).expect("tokenizer serializes");
        let restored: Tokenizer = bincode::deserialize(&bytes).expect("tokenizer deserializes");
        let start = restored.group_dfa_start_state(1);
        let end = restored
            .group_dfa_step(1, start, b'x')
            .expect("isolated group DFA rebuilds after deserialization");
        assert!(restored.group_dfa_is_match(1, end));
    }

    #[test]
    fn legacy_tokenizer_wire_rebuilds_group_dfas() {
        let expressions = vec![Expr::U8Seq(b"ab".to_vec()), Expr::U8Seq(b"x".to_vec())];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let legacy = LegacyTokenizerWire {
            dfa: tokenizer.dfa.clone(),
            num_terminals: tokenizer.num_terminals,
        };

        let bytes = bincode::serialize(&legacy).expect("legacy tokenizer serializes");
        let restored: Tokenizer = bincode::deserialize(&bytes).expect("legacy tokenizer deserializes");
        assert_eq!(restored.group_dfa_count(), 2);
        let end = restored
            .group_dfa_step(0, restored.group_dfa_start_state(0), b'a')
            .and_then(|state| restored.group_dfa_step(0, state, b'b'))
            .expect("restored group DFA recognizes ab");
        assert!(restored.group_dfa_is_match(0, end));
    }

    #[test]
    fn group_dfa_cache_is_invalidated_when_default_entry_changes() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, b'a', 2);
        dfa.add_transition(1, b'b', 3);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(2, finalizers.clone(), BitSet::new(1));
        dfa.overwrite_state_metadata(3, finalizers, BitSet::new(1));
        dfa.set_start_states(vec![0, 1]);
        let mut tokenizer = Tokenizer::from_parts(dfa, 1, None);

        let first_start = tokenizer.group_dfa_start_state(0);
        assert!(tokenizer
            .group_dfa_step(0, first_start, b'a')
            .is_some());
        assert_eq!(tokenizer.group_dfa_step(0, first_start, b'b'), None);

        tokenizer.set_default_start_state(1);
        let second_start = tokenizer.group_dfa_start_state(0);
        assert_eq!(tokenizer.group_dfa_step(0, second_start, b'a'), None);
        assert!(tokenizer
            .group_dfa_step(0, second_start, b'b')
            .is_some());
    }

    #[test]
    fn group_dfas_preserve_generic_selectable_entries() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(1);
        dfa.add_transition(0, b'a', 2);
        dfa.add_transition(1, b'b', 3);
        let mut finalizers = BitSet::new(1);
        finalizers.set(0);
        dfa.overwrite_state_metadata(2, finalizers.clone(), BitSet::new(1));
        dfa.overwrite_state_metadata(3, finalizers, BitSet::new(1));
        dfa.set_start_states(vec![0, 1]);
        dfa.recompute_possible_futures();
        let tokenizer = Tokenizer::from_parts(dfa, 1, None);

        let group = tokenizer.group_dfa(0);
        assert_eq!(group.dfa.start_states().len(), 2);
        let default_end = group
            .dfa
            .step(group.dfa.start_state(), b'a')
            .expect("default isolated entry accepts a");
        let auxiliary_end = group
            .dfa
            .step(group.dfa.start_states()[1], b'b')
            .expect("auxiliary isolated entry accepts b");
        assert!(group.is_match(default_end));
        assert!(group.is_match(auxiliary_end));
    }

}
