//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::automata::dfa::DFA;
use crate::automata::regex::Expr;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondaryLexer {
    pub(crate) dfa: DFA,
    pub(crate) terminal_to_guard: Vec<Option<u32>>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondaryVirtualStateSpace {
    pub(crate) main_representatives: Vec<u32>,
    pub(crate) main_state_to_rep_index: Vec<u32>,
    pub(crate) secondary_representatives: Vec<u32>,
    pub(crate) secondary_state_to_rep_index: Vec<u32>,
    pub(crate) dead_rep_index: u32,
    pub(crate) product_pairs: Vec<(u32, u32)>,
    pub(crate) pair_to_state: BTreeMap<(u32, u32), u32>,
}

impl SecondaryVirtualStateSpace {
    #[inline]
    pub(crate) fn state_id_for_pair(&self, main_rep_index: u32, secondary_rep_index: u32) -> u32 {
        self.pair_to_state
            .get(&(main_rep_index, secondary_rep_index))
            .copied()
            .unwrap_or(u32::MAX)
    }

    #[inline]
    pub(crate) fn decode(&self, state: u32) -> (u32, u32) {
        self.product_pairs[state as usize]
    }

    #[inline]
    pub(crate) fn state_count(&self) -> u32 {
        self.product_pairs.len() as u32
    }

    #[inline]
    pub(crate) fn rep_index_for_main_state(&self, main_state: u32) -> u32 {
        self.main_state_to_rep_index[main_state as usize]
    }

    #[inline]
    pub(crate) fn rep_index_for_secondary_state(&self, secondary_state: u32) -> u32 {
        if secondary_state == u32::MAX {
            self.dead_rep_index
        } else {
            self.secondary_state_to_rep_index[secondary_state as usize]
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub(crate) dfa: DFA,
    pub num_terminals: u32,
    /// Optional secondary guard DFA. Guard finalizer ids are guard ids;
    /// terminal_to_guard maps primary terminal ids to guard ids.
    #[serde(default)]
    pub(crate) secondary: Option<Arc<SecondaryLexer>>,
    #[serde(default)]
    pub(crate) secondary_virtual: Option<Arc<SecondaryVirtualStateSpace>>,
    /// Per-terminal regex expressions used to (re)build this tokenizer.
    /// Skipped during (de)serialization because they are only needed during
    /// compile-time simplification for active-terminal rebuilds.
    #[serde(default, skip)]
    pub(crate) exprs: Option<Arc<[Expr]>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMatch {
    pub id: TerminalID,
    pub width: usize,
    pub end_state: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExecResult {
    pub end_state: Option<usize>,
    pub matches: Vec<TokenizerMatch>,
}

fn into_longest_matches(matches: FxHashMap<TerminalID, (usize, usize)>) -> Vec<TokenizerMatch> {
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
    pub fn start_state(&self) -> u32 {
        0
    }

    pub(crate) fn has_secondary(&self) -> bool {
        self.secondary.is_some()
    }

    pub(crate) fn has_secondary_virtual_state_space(&self) -> bool {
        self.secondary_virtual.is_some()
    }

    pub(crate) fn set_secondary_virtual_state_space(
        &mut self,
        main_representatives: Vec<u32>,
        main_state_to_rep_index: Vec<u32>,
        secondary_representatives: Vec<u32>,
        secondary_state_to_rep_index: Vec<u32>,
        dead_rep_index: u32,
        product_pairs: Vec<(u32, u32)>,
        pair_to_state: BTreeMap<(u32, u32), u32>,
    ) {
        self.secondary_virtual = Some(Arc::new(SecondaryVirtualStateSpace {
            main_representatives,
            main_state_to_rep_index,
            secondary_representatives,
            secondary_state_to_rep_index,
            dead_rep_index,
            product_pairs,
            pair_to_state,
        }));
    }

    pub(crate) fn virtual_original_state_for_runtime(&self, state: usize) -> usize {
        let Some(space) = &self.secondary_virtual else {
            return state;
        };
        let main = Self::runtime_primary_state(state);
        let secondary = Self::runtime_secondary_state(state);
        let main_rep_index = space.rep_index_for_main_state(main);
        let secondary_rep_index = space.rep_index_for_secondary_state(secondary);
        space.state_id_for_pair(main_rep_index, secondary_rep_index) as usize
    }

    pub(crate) fn decode_virtual_original_state(&self, state: u32) -> Option<(u32, u32)> {
        let space = self.secondary_virtual.as_ref()?;
        let (main_rep_index, secondary_rep_index) = space.decode(state);
        let main = space.main_representatives[main_rep_index as usize];
        let secondary = space.secondary_representatives[secondary_rep_index as usize];
        Some((main, secondary))
    }

    #[inline]
    pub(crate) fn pack_runtime_state(main: u32, secondary: u32) -> usize {
        ((secondary as usize) << 32) | main as usize
    }

    #[inline]
    pub(crate) fn runtime_primary_state(state: usize) -> u32 {
        (state & 0xffff_ffff) as u32
    }

    #[inline]
    pub(crate) fn runtime_secondary_state(state: usize) -> u32 {
        (state >> 32) as u32
    }

    pub(crate) fn step_runtime_state(&self, state: usize, byte: u8) -> Option<usize> {
        let main = Self::runtime_primary_state(state);
        let next_main = self.step(main, byte)?;
        let Some(secondary) = &self.secondary else {
            return Some(next_main as usize);
        };
        let secondary_state = Self::runtime_secondary_state(state);
        let next_secondary = if secondary_state == u32::MAX {
            u32::MAX
        } else {
            secondary.dfa.step(secondary_state, byte).unwrap_or(u32::MAX)
        };
        Some(Self::pack_runtime_state(next_main, next_secondary))
    }

    /// Detect nullable terminals (those that match the empty string) by
    /// inspecting start-state finalizers, remove them from the DFA, and return
    /// the set.  After this call the tokenizer no longer reports those
    /// terminals as matched at state 0.
    pub fn isolate_start_state_and_drain_nullable_terminals(&mut self) -> BTreeSet<TerminalID> {
        self.isolate_start_state();
        self.dfa
            .clear_finalizers_for_state(self.start_state())
            .iter()
            .map(|terminal| terminal as TerminalID)
            .collect()
    }

    /// Ensure that no byte transition in the DFA targets the start state.
    ///
    /// If any transition does, a copy of the start state is created and all
    /// such transitions are redirected to the copy.  This keeps the DFA
    /// equivalent while guaranteeing the start state is only reachable at
    /// position 0.
    fn isolate_start_state(&mut self) {
        let start = self.start_state();
        if !self.has_incoming_start_transitions(start) {
            return;
        }
        let clone_id = self.dfa.clone_state(start);
        self.dfa.redirect_transitions(start, clone_id);
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    pub fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.dfa.get_transition(state, byte)
    }

    pub(crate) fn original_state_transition(&self, state: u32, byte: u8) -> u32 {
        let Some(space) = &self.secondary_virtual else {
            return self.dfa.get_transition(state, byte);
        };
        let Some(secondary) = &self.secondary else {
            return self.dfa.get_transition(state, byte);
        };
        let (main_rep_index, secondary_rep_index) = space.decode(state);
        let main = space.main_representatives[main_rep_index as usize];
        let next_main = self.dfa.get_transition(main, byte);
        if next_main == u32::MAX {
            return u32::MAX;
        }
        let next_main_rep_index = space.rep_index_for_main_state(next_main);
        let secondary_rep_state = space.secondary_representatives[secondary_rep_index as usize];
        let next_secondary_rep_index = if secondary_rep_state == u32::MAX {
            space.dead_rep_index
        } else {
            secondary
                .dfa
                .step(secondary_rep_state, byte)
                .map(|s| space.secondary_state_to_rep_index[s as usize])
                .unwrap_or(space.dead_rep_index)
        };
        space.state_id_for_pair(next_main_rep_index, next_secondary_rep_index)
    }
    pub fn run(&self, input: &[u8]) -> u32 {
        input
            .iter()
            .try_fold(self.start_state(), |state, &byte| self.step(state, byte))
            .unwrap_or(self.start_state())
    }

    pub(crate) fn original_state_finalizers(&self, state: u32) -> BitSet {
        let Some(space) = &self.secondary_virtual else {
            return self.dfa.finalizers(state).clone();
        };
        let Some(secondary) = &self.secondary else {
            return self.dfa.finalizers(state).clone();
        };
        let (main_rep_index, secondary_rep_index) = space.decode(state);
        let main = space.main_representatives[main_rep_index as usize];
        let secondary_rep_state = space.secondary_representatives[secondary_rep_index as usize];
        let mut out = BitSet::new(self.num_terminals as usize);
        for terminal in self.dfa.finalizers(main).iter() {
            match secondary.terminal_to_guard.get(terminal).copied().flatten() {
                None => out.set(terminal),
                Some(guard) => {
                    if secondary_rep_state != u32::MAX
                        && secondary.dfa.finalizers(secondary_rep_state).contains(guard as usize)
                    {
                        out.set(terminal);
                    }
                }
            }
        }
        out
    }
    pub(crate) fn original_state_possible_futures(&self, state: u32) -> BitSet {
        let Some(space) = &self.secondary_virtual else {
            return self.dfa.possible_future_group_ids(state).clone();
        };
        let Some(secondary) = &self.secondary else {
            return self.dfa.possible_future_group_ids(state).clone();
        };
        let (main_rep_index, secondary_rep_index) = space.decode(state);
        let main = space.main_representatives[main_rep_index as usize];
        let secondary_rep_state = space.secondary_representatives[secondary_rep_index as usize];
        let mut out = self.dfa.possible_future_group_ids(main).clone();
        for terminal in 0..self.num_terminals as usize {
            let Some(guard) = secondary.terminal_to_guard.get(terminal).copied().flatten() else {
                continue;
            };
            let possible = secondary_rep_state != u32::MAX
                && secondary
                    .dfa
                    .possible_future_group_ids(secondary_rep_state)
                    .contains(guard as usize);
            if !possible && terminal < out.len() {
                out.clear(terminal);
            }
        }
        out
    }
    pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.matched_terminals_iter(state).collect()
    }

    pub(crate) fn matched_terminals_runtime(&self, state: usize) -> Vec<TerminalID> {
        let main = Self::runtime_primary_state(state);
        let Some(secondary) = &self.secondary else {
            return self.matched_terminals_iter(main).collect();
        };
        let secondary_state = Self::runtime_secondary_state(state);
        self.dfa
            .finalizers(main)
            .iter()
            .filter_map(|terminal| {
                let terminal = terminal as TerminalID;
                match secondary.terminal_to_guard.get(terminal as usize).copied().flatten() {
                    None => Some(terminal),
                    Some(guard) => {
                        if secondary_state != u32::MAX
                            && secondary.dfa.finalizers(secondary_state).contains(guard as usize)
                        {
                            Some(terminal)
                        } else {
                            None
                        }
                    }
                }
            })
            .collect()
    }

    pub(crate) fn possible_future_terminals_runtime(&self, state: usize) -> BitSet {
        let main = Self::runtime_primary_state(state);
        let mut futures = self.dfa.possible_future_group_ids(main).clone();
        let Some(secondary) = &self.secondary else {
            return futures;
        };
        let secondary_state = Self::runtime_secondary_state(state);
        for terminal in 0..self.num_terminals as usize {
            let Some(guard) = secondary.terminal_to_guard.get(terminal).copied().flatten() else {
                continue;
            };
            let guard_possible = secondary_state != u32::MAX
                && secondary
                    .dfa
                    .possible_future_group_ids(secondary_state)
                    .contains(guard as usize);
            if !guard_possible && terminal < futures.len() {
                futures.clear(terminal);
            }
        }
        futures
    }

    pub(crate) fn is_end_runtime(&self, state: usize) -> bool {
        self.possible_future_terminals_runtime(state).is_empty()
    }

    pub(crate) fn matched_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.dfa
            .finalizers(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    pub(crate) fn possible_future_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.dfa
            .possible_future_group_ids(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    pub fn possible_future_terminals(&self, state: u32) -> &BitSet {
        self.dfa.possible_future_group_ids(state)
    }

    pub fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    pub fn num_states(&self) -> u32 {
        self.secondary_virtual
            .as_ref()
            .map_or(self.dfa.num_states() as u32, |space| space.state_count())
    }

    pub(crate) fn num_forced_minimized_states(&self) -> usize {
        self.dfa.minimize().num_states()
    }

    pub(crate) fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: usize,
    ) -> TokenizerExecResult {
        let mut matches = Vec::new();
        let end_state = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_all_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state: end_state.filter(|&state| !self.is_end_runtime(state)),
            matches,
        }
    }

    pub fn execute_from_state(&self, input: &[u8], start: usize) -> TokenizerExecResult {
        let mut matches = FxHashMap::<TerminalID, (usize, usize)>::default();
        let end_state = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_longest_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state,
            matches: into_longest_matches(matches),
        }
    }

    pub(crate) fn execute_from_state_end_only(&self, input: &[u8], start: usize) -> Option<usize> {
        self.scan_input(input, start, &mut (), |_, _, _, _| {})
    }

    pub fn execute_all_matches(&self, input: &[u8], start: usize) -> TokenizerResult {
        let exec = self.execute_from_state_all_widths(input, start);
        let end_state = exec.end_state.unwrap_or(start);
        TokenizerResult {
            end_state,
            matches: group_matches_by_width(exec.matches),
        }
    }

    pub fn initial_state(&self) -> usize {
        if self.secondary.is_some() {
            Self::pack_runtime_state(self.start_state(), self.start_state())
        } else {
            self.start_state() as usize
        }
    }

    pub fn initial_state_id(&self) -> u32 {
        self.start_state()
    }

    pub fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
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
    pub fn scan_terminal_matches_from_state(
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

    fn has_incoming_start_transitions(&self, start: u32) -> bool {
        self.dfa
            .states()
            .iter()
            .any(|state| state.transitions.values().any(|&target| target == start))
    }

    fn record_all_matches(&self, matches: &mut Vec<TokenizerMatch>, state: usize, width: usize) {
        matches.extend(self.matched_terminals_runtime(state).into_iter().map(|id| TokenizerMatch {
            id,
            width,
            end_state: state,
        }));
    }

    fn record_longest_matches(
        &self,
        matches: &mut FxHashMap<TerminalID, (usize, usize)>,
        state: usize,
        width: usize,
    ) {
        for terminal in self.matched_terminals_runtime(state) {
            matches.insert(terminal, (width, state));
        }
    }

    fn scan_input<R>(
        &self,
        input: &[u8],
        start: usize,
        mut matches: &mut R,
        mut record_matches: impl FnMut(&Self, &mut R, usize, usize),
    ) -> Option<usize> {
        let mut state = start;
        for (index, &byte) in input.iter().enumerate() {
            let next = self.step_runtime_state(state, byte)?;
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
                    secondary: self.secondary.clone(),
                    secondary_virtual: self.secondary_virtual.clone(),
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
                        secondary: self.secondary.clone(),
                        secondary_virtual: self.secondary_virtual.clone(),
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
                    secondary: self.secondary.clone(),
                    secondary_virtual: self.secondary_virtual.clone(),
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
                secondary: self.secondary.clone(),
                secondary_virtual: self.secondary_virtual.clone(),
                exprs: self.exprs.clone(),
            },
            ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                state_mapping,
                post_minimize_states as u32,
            ),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    pub end_state: usize,
    pub matches: Vec<(usize, BTreeSet<TerminalID>)>,
}
