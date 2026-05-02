//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::BTreeSet;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::automata::dfa::DFA;
use crate::automata::regex::Expr;
use crate::grammar::flat::TerminalID;
use crate::ds::bitset::BitSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub(crate) dfa: DFA,
    pub num_terminals: u32,
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
    pub end_state: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExecResult {
    pub end_state: Option<u32>,
    pub matches: Vec<TokenizerMatch>,
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

impl Tokenizer {
    pub fn start_state(&self) -> u32 {
        0
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

    pub fn run(&self, input: &[u8]) -> u32 {
        input
            .iter()
            .try_fold(self.start_state(), |state, &byte| self.step(state, byte))
            .unwrap_or(self.start_state())
    }

    pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.matched_terminals_iter(state).collect()
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

    pub fn all_matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.matched_terminals(state)
    }

    pub fn possible_future_terminals(&self, state: u32) -> &BitSet {
        self.dfa.possible_future_group_ids(state)
    }

    pub fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    pub fn num_states(&self) -> u32 {
        self.dfa.num_states() as u32
    }

    pub(crate) fn execute_from_state_all_widths(
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

    pub fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {
        let mut matches = FxHashMap::<TerminalID, (usize, u32)>::default();
        let end_state = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_longest_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state,
            matches: into_longest_matches(matches),
        }
    }

    pub(crate) fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> Option<u32> {
        self.scan_input(input, start, &mut (), |_, _, _, _| {})
    }

    pub fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
        let exec = self.execute_from_state_all_widths(input, start);
        let end_state = exec.end_state.unwrap_or(start);
        TokenizerResult {
            end_state,
            matches: group_matches_by_width(exec.matches),
        }
    }

    pub fn initial_state(&self) -> u32 {
        self.start_state()
    }

    pub fn initial_state_id(&self) -> u32 {
        self.initial_state()
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    pub end_state: u32,
    pub matches: Vec<(usize, BTreeSet<TerminalID>)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::bytes;
    use crate::automata::lexer::regex::parse_regex;
    use crate::compiler::compile::build_tokenizer_from_exprs;

    #[test]
    fn test_execute_from_state_keeps_only_longest_match_per_terminal() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"aa")]);

        let exec = tokenizer.execute_from_state(b"aa", tokenizer.start_state());

        assert_eq!(
            exec.matches,
            vec![
                TokenizerMatch {
                    id: 0,
                    width: 1,
                    end_state: tokenizer.run(b"a"),
                },
                TokenizerMatch {
                    id: 1,
                    width: 2,
                    end_state: tokenizer.run(b"aa"),
                },
            ]
        );
    }

    #[test]
    fn test_execute_from_state_replaces_shorter_match_for_same_terminal() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), parse_regex("a+", true)]);

        let exec = tokenizer.execute_from_state(b"aa", tokenizer.start_state());

        assert_eq!(
            exec.matches,
            vec![
                TokenizerMatch {
                    id: 0,
                    width: 1,
                    end_state: tokenizer.run(b"a"),
                },
                TokenizerMatch {
                    id: 1,
                    width: 2,
                    end_state: tokenizer.run(b"aa"),
                },
            ]
        );
    }

    #[test]
    fn test_execute_all_matches_keeps_all_widths() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), parse_regex("a+", true)]);

        let result = tokenizer.execute_all_matches(b"aa", tokenizer.start_state());

        assert_eq!(
            result.matches,
            vec![
                (1, BTreeSet::from([0, 1])),
                (2, BTreeSet::from([1])),
            ]
        );
    }

    #[test]
    fn test_scan_terminal_matches_ignores_initial_state_finalizer() {
        // Terminal 0 matches "" or "a"; terminal 1 matches "b".
        let tokenizer = build_tokenizer_from_exprs(&[
            parse_regex("a?", true),
            bytes(b"b"),
        ]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"", tokenizer.start_state(), &interest);
        assert!(matched.is_empty(), "initial state finalizer should be ignored");
        assert!(
            end_state.is_some(),
            "futures should still overlap since terminal 0 can match 'a'"
        );
    }

    #[test]
    fn test_scan_terminal_matches_returns_matched_terminals_after_bytes() {
        // Terminal 0 = "a", terminal 1 = "aa".
        // After consuming "a", terminal 0 matched and terminal 1 remains
        // with a future (can continue with another "a").
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"aa")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        interest.set(1);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"a", tokenizer.start_state(), &interest);
        assert!(matched.contains(0), "terminal 0 ('a') should match");
        assert!(!matched.contains(1), "terminal 1 ('aa') should not match yet");
        assert!(end_state.is_some(), "end state should exist because terminal 1 still has a future");
    }

    #[test]
    fn test_scan_terminal_matches_returns_none_when_futures_diverge() {
        // Terminal 0 = "a", terminal 1 = "b".
        // After consuming "a", terminal 1 has no future from the end state.
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"b")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        interest.set(1);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"a", tokenizer.start_state(), &interest);
        // Terminal 0 matched, terminal 1 remained in `remaining` but has no future.
        assert!(matched.contains(0));
        assert!(!matched.contains(1));
        assert!(
            end_state.is_none(),
            "end state should be None because futures of end state do not overlap remaining"
        );
    }

    #[test]
    fn test_scan_terminal_matches_returns_some_when_end_state_still_has_future() {
        // Terminal 0 = "a", terminal 1 = "ab".
        // After consuming "a", terminal 0 matched. Terminal 1 remains and
        // the end state still has a future for terminal 1 (can continue with 'b').
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"ab")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        interest.set(1);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"a", tokenizer.start_state(), &interest);
        assert!(matched.contains(0), "terminal 0 should match");
        assert!(!matched.contains(1), "terminal 1 should not have matched yet");
        assert!(
            end_state.is_some(),
            "end state should be Some because terminal 1 still has a future"
        );
    }

    #[test]
    fn test_scan_terminal_matches_respects_terminals_of_interest_filter() {
        // Terminal 0 = "a", terminal 1 = "b", terminal 2 = "c".
        // We only care about terminal 1 ("b"). Input "a" does not match it,
        // and from the state-after-a there is no future for "b".
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"b"), bytes(b"c")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(1);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"a", tokenizer.start_state(), &interest);
        assert!(matched.is_empty(), "no terminal of interest should have matched");
        assert!(
            end_state.is_none(),
            "end state should be None because no future for terminal 1 after 'a'"
        );
    }

    #[test]
    fn test_scan_terminal_matches_early_stop_on_no_transition() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"ab", tokenizer.start_state(), &interest);
        // After "a", terminal 0 matched. Then on "b" there is no transition.
        assert!(matched.contains(0));
        assert!(end_state.is_none());
    }
}
