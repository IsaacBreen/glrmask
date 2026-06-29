//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::BTreeSet;
use std::sync::Arc;

use rustc_hash::FxHashMap;
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

/// For each original tokenizer state and terminal, the bytes whose transition
/// stays in that terminal's minimized residual-language class.
///
/// The row index is an original tokenizer state; the column index is a terminal.
pub(crate) type TerminalSelfLoopBytes = Arc<[Box<[U8Set]>]>;

#[inline]
fn ordered_pair(left: u32, right: u32) -> (u32, u32) {
    if left < right { (left, right) } else { (right, left) }
}

#[inline]
fn or_words(into: &mut [u64], other: &[u64]) {
    for (into, other) in into.iter_mut().zip(other) {
        *into |= *other;
    }
}

fn ensure_pair(
    pairs: &mut Vec<(u32, u32)>,
    pair_indices: &mut FxHashMap<(u32, u32), usize>,
    bases: &mut Vec<Vec<u64>>,
    successors: &mut Vec<Vec<usize>>,
    left: u32,
    right: u32,
    num_words: usize,
) -> usize {
    debug_assert_ne!(left, right);
    let pair = ordered_pair(left, right);
    if let Some(&index) = pair_indices.get(&pair) {
        return index;
    }
    let index = pairs.len();
    pairs.push(pair);
    pair_indices.insert(pair, index);
    bases.push(vec![0; num_words]);
    successors.push(Vec::new());
    index
}

/// Return the terminal-wise distinction bitsets for every product pair.
///
/// For a terminal `t`, two states are equivalent precisely when the DFA formed
/// by retaining only `t`'s finalizers would merge them.  A product node `(p,q)`
/// therefore carries the terminals that distinguish `p` from `q`.  Its local
/// contribution is their finalizer xor, plus every terminal for a byte present
/// on only one side; common byte transitions recurse to the corresponding
/// product node.  Solving those monotone equations over product-graph SCCs is
/// equivalent to minimizing once per terminal, but shares all transition work.
fn compute_terminal_self_loop_bytes(tokenizer: &Tokenizer) -> TerminalSelfLoopBytes {
    let num_states = tokenizer.dfa.num_states();
    let num_terminals = tokenizer.num_terminals as usize;
    let num_words = num_terminals.div_ceil(64);
    let mut rows = (0..num_states)
        .map(|_| vec![U8Set::empty(); num_terminals].into_boxed_slice())
        .collect::<Vec<_>>();

    if num_states == 0 || num_terminals == 0 {
        return Arc::from(rows.into_boxed_slice());
    }

    let mut all_terminals = vec![u64::MAX; num_words];
    if let Some(last) = all_terminals.last_mut() {
        let remainder = num_terminals % 64;
        if remainder != 0 {
            *last = (1u64 << remainder) - 1;
        }
    }

    let mut pairs = Vec::<(u32, u32)>::new();
    let mut pair_indices = FxHashMap::<(u32, u32), usize>::default();
    let mut bases = Vec::<Vec<u64>>::new();
    let mut successors = Vec::<Vec<usize>>::new();

    // Only pairs reachable from an original transition can be needed when
    // deciding whether that transition is a quotient self-loop.
    for (state, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        for (_, &target) in dfa_state.transitions.iter() {
            if target != state as u32 {
                ensure_pair(
                    &mut pairs,
                    &mut pair_indices,
                    &mut bases,
                    &mut successors,
                    state as u32,
                    target,
                    num_words,
                );
            }
        }
    }

    let mut pair_index = 0usize;
    while pair_index < pairs.len() {
        let (left, right) = pairs[pair_index];
        let left_state = &tokenizer.dfa.states()[left as usize];
        let right_state = &tokenizer.dfa.states()[right as usize];

        for ((base, left_finalizers), right_finalizers) in bases[pair_index]
            .iter_mut()
            .zip(left_state.finalizers.words())
            .zip(right_state.finalizers.words())
        {
            *base |= left_finalizers ^ right_finalizers;
        }

        let mut left_transitions = left_state.transitions.iter().peekable();
        let mut right_transitions = right_state.transitions.iter().peekable();
        while left_transitions.peek().is_some() || right_transitions.peek().is_some() {
            match (left_transitions.peek().copied(), right_transitions.peek().copied()) {
                (Some((left_byte, left_target)), Some((right_byte, right_target)))
                    if left_byte == right_byte => {
                    left_transitions.next();
                    right_transitions.next();
                    if left_target != right_target {
                        let successor = ensure_pair(
                            &mut pairs,
                            &mut pair_indices,
                            &mut bases,
                            &mut successors,
                            *left_target,
                            *right_target,
                            num_words,
                        );
                        successors[pair_index].push(successor);
                    }
                }
                (Some((left_byte, _)), Some((right_byte, _))) if left_byte < right_byte => {
                    left_transitions.next();
                    or_words(&mut bases[pair_index], &all_terminals);
                }
                (Some(_), Some(_)) => {
                    right_transitions.next();
                    or_words(&mut bases[pair_index], &all_terminals);
                }
                (Some(_), None) => {
                    left_transitions.next();
                    or_words(&mut bases[pair_index], &all_terminals);
                }
                (None, Some(_)) => {
                    right_transitions.next();
                    or_words(&mut bases[pair_index], &all_terminals);
                }
                (None, None) => break,
            }
        }

        successors[pair_index].sort_unstable();
        successors[pair_index].dedup();
        pair_index += 1;
    }

    if pairs.is_empty() {
        return Arc::from(rows.into_boxed_slice());
    }

    // Kosaraju SCC decomposition of the product graph.  Each SCC has one
    // shared distinction set, then sinks are solved before their predecessors.
    let mut reverse = vec![Vec::<usize>::new(); pairs.len()];
    for (source, targets) in successors.iter().enumerate() {
        for &target in targets {
            reverse[target].push(source);
        }
    }

    let mut seen = vec![false; pairs.len()];
    let mut post_order = Vec::with_capacity(pairs.len());
    for root in 0..pairs.len() {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![(root, 0usize)];
        while let Some((node, edge_index)) = stack.last_mut() {
            if *edge_index < successors[*node].len() {
                let next = successors[*node][*edge_index];
                *edge_index += 1;
                if !seen[next] {
                    seen[next] = true;
                    stack.push((next, 0));
                }
            } else {
                post_order.push(*node);
                stack.pop();
            }
        }
    }

    let mut scc_of_pair = vec![usize::MAX; pairs.len()];
    let mut scc_count = 0usize;
    for &root in post_order.iter().rev() {
        if scc_of_pair[root] != usize::MAX {
            continue;
        }
        scc_of_pair[root] = scc_count;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            for &previous in &reverse[node] {
                if scc_of_pair[previous] == usize::MAX {
                    scc_of_pair[previous] = scc_count;
                    stack.push(previous);
                }
            }
        }
        scc_count += 1;
    }

    let mut scc_distinctions = vec![vec![0u64; num_words]; scc_count];
    let mut scc_successors = vec![Vec::<usize>::new(); scc_count];
    for (pair, &scc) in scc_of_pair.iter().enumerate() {
        or_words(&mut scc_distinctions[scc], &bases[pair]);
        for &successor in &successors[pair] {
            let successor_scc = scc_of_pair[successor];
            if successor_scc != scc {
                scc_successors[scc].push(successor_scc);
            }
        }
    }
    for targets in &mut scc_successors {
        targets.sort_unstable();
        targets.dedup();
    }

    let mut scc_predecessors = vec![Vec::<usize>::new(); scc_count];
    let mut remaining_successors = vec![0usize; scc_count];
    for (source, targets) in scc_successors.iter().enumerate() {
        remaining_successors[source] = targets.len();
        for &target in targets {
            scc_predecessors[target].push(source);
        }
    }

    let mut ready = remaining_successors
        .iter()
        .enumerate()
        .filter_map(|(scc, &remaining)| (remaining == 0).then_some(scc))
        .collect::<Vec<_>>();
    let mut ready_index = 0usize;
    while ready_index < ready.len() {
        let solved = ready[ready_index];
        ready_index += 1;
        let solved_distinctions = scc_distinctions[solved].clone();
        for &predecessor in &scc_predecessors[solved] {
            or_words(
                &mut scc_distinctions[predecessor],
                &solved_distinctions,
            );
            remaining_successors[predecessor] -= 1;
            if remaining_successors[predecessor] == 0 {
                ready.push(predecessor);
            }
        }
    }
    debug_assert_eq!(ready.len(), scc_count);

    for (state, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        for (byte, &target) in dfa_state.transitions.iter() {
            if target == state as u32 {
                for terminal in &mut rows[state] {
                    terminal.insert(byte);
                }
                continue;
            }
            let pair = ordered_pair(state as u32, target);
            let pair_index = pair_indices[&pair];
            let distinctions = &scc_distinctions[scc_of_pair[pair_index]];
            for (terminal, loop_bytes) in rows[state].iter_mut().enumerate() {
                if distinctions[terminal / 64] & (1u64 << (terminal % 64)) == 0 {
                    loop_bytes.insert(byte);
                }
            }
        }
    }

    Arc::from(rows.into_boxed_slice())
}

impl Tokenizer {
    /// Compute the terminal-sensitive quotient self-loop map used by direct
    /// dynamic masking.  This is intentionally uncached here: the runtime-only
    /// dynamic-mask artifact owns the lazy cache so tokenizer clones and DFA
    /// simplifications never inherit stale data.
    pub(crate) fn terminal_self_loop_bytes_map(&self) -> TerminalSelfLoopBytes {
        compute_terminal_self_loop_bytes(self)
    }

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

    fn start_state(&self) -> u32 {
        0
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
        let clone_id = self.dfa.clone_state(start);
        if !self.dfa.redirect_transitions(start, clone_id) {
            self.dfa.discard_last_state(clone_id);
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

    /// Clear inactive terminal metadata while preserving every DFA state ID and
    /// byte transition. The terminal-interchangeability reference path needs a
    /// stable original-state coordinate for its transport maps, so it cannot use
    /// the minimizing simplifier below.
    pub(crate) fn deactivate_terminals_without_minimizing(
        &self,
        active_terminals: &[bool],
    ) -> Tokenizer {
        assert_eq!(active_terminals.len(), self.num_terminals as usize);
        let mut dfa = self.dfa.clone();
        for state in 0..dfa.num_states() as u32 {
            let mut finalizers = BitSet::new(self.num_terminals as usize);
            for terminal in dfa.finalizers(state).iter() {
                if active_terminals[terminal] {
                    finalizers.set(terminal);
                }
            }
            let mut futures = BitSet::new(self.num_terminals as usize);
            for terminal in dfa.possible_future_group_ids(state).iter() {
                if active_terminals[terminal] {
                    futures.set(terminal);
                }
            }
            dfa.overwrite_state_metadata(state, finalizers, futures);
        }
        Tokenizer {
            dfa,
            num_terminals: self.num_terminals,
            exprs: self.exprs.clone(),
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
mod terminal_self_loop_tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::compile::build_regex;
    use crate::ds::u8set::U8Set;

    fn repeat(bytes: &[u8]) -> Expr {
        Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(bytes))),
            min: 1,
            max: None,
        }
    }

    #[test]
    fn terminal_self_loop_map_matches_per_terminal_minimization() {
        let exprs = vec![
            repeat(b"ab"),
            Expr::Seq(vec![
                Expr::U8Seq(b"a".to_vec()),
                repeat(b"bc"),
            ]),
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"ac".to_vec()),
                Expr::U8Seq(b"abc".to_vec()),
            ]),
            Expr::Seq(vec![
                Expr::U8Seq(b"x".to_vec()),
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"y".to_vec())),
                    min: 0,
                    max: Some(3),
                },
            ]),
        ];
        let tokenizer = build_regex(&exprs).into_tokenizer(
            exprs.len() as u32,
            Some(Arc::from(exprs.into_boxed_slice())),
        );
        let loops = tokenizer.terminal_self_loop_bytes_map();

        for terminal in 0..tokenizer.num_terminals as usize {
            let mut terminal_only = tokenizer.dfa.clone();
            for state in terminal_only.states_mut().iter_mut() {
                for other_terminal in 0..tokenizer.num_terminals as usize {
                    if other_terminal != terminal {
                        state.finalizers.clear(other_terminal);
                    }
                }
            }
            let (_, state_class) = terminal_only.minimize_with_state_mapping_preserve_all_states();

            for (state, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
                for (byte, &target) in dfa_state.transitions.iter() {
                    let expected = state_class[state] == state_class[target as usize];
                    assert_eq!(
                        loops[state][terminal].contains(byte),
                        expected,
                        "terminal={terminal} state={state} byte={byte:#04x} target={target}",
                    );
                }
            }
        }
    }
}
