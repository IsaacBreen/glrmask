//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

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
        }
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
    }

    /// Select the default lexer entry state while retaining all other entries.
    pub fn set_default_start_state(&mut self, state: u32) {
        self.dfa.set_default_start_state(state);
    }

    /// Add an auxiliary selectable lexer entry state.
    pub fn add_start_state(&mut self, state: u32) {
        self.dfa.add_start_state(state);
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
                Tokenizer {
                    dfa,
                    num_terminals: self.num_terminals,
                    exprs: self.exprs.clone(),
                },
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
                    Tokenizer {
                        dfa,
                        num_terminals: self.num_terminals,
                        exprs: self.exprs.clone(),
                    },
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
                Tokenizer {
                    dfa,
                    num_terminals: self.num_terminals,
                    exprs: self.exprs.clone(),
                },
                identity,
            );
        }

        let (minimized, state_mapping) = dfa.minimize_with_state_mapping();
        let post_minimize_states = minimized.num_states();

        (
            Tokenizer {
                dfa: minimized,
                num_terminals: self.num_terminals,
                exprs: self.exprs.clone(),
            },
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
        Tokenizer {
            dfa: self.relabel_dfa_for_terminal_labels(terminal_labels, active_labels),
            num_terminals: self.num_terminals,
            exprs: self.exprs.clone(),
        }
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

}
