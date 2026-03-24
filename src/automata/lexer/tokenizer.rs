//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::BTreeSet;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::automata::dfa::DFA;
use crate::compiler::grammar_def::TerminalID;
use crate::ds::bitset::BitSet;

pub use super::regex::parse_regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub(crate) dfa: DFA,
    pub num_terminals: u32,
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
        let has_incoming = self.dfa.states().iter().any(|st| {
            st.transitions.values().any(|&target| target == start)
        });
        if !has_incoming {
            return;
        }
        // Clone the start state as a new state.
        let clone_id = self.dfa.clone_state(start);
        // Redirect every transition that points to start → clone.
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
        self.dfa
            .finalizers(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
            .collect()
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
        let mut state = start;
        let mut matches = Vec::new();

        for (index, &byte) in input.iter().enumerate() {
            let Some(next) = self.step(state, byte) else {
                return TokenizerExecResult {
                    end_state: None,
                    matches,
                };
            };
            state = next;
            for terminal in self.dfa.finalizers(state).iter() {
                matches.push(TokenizerMatch {
                    id: terminal as TerminalID,
                    width: index + 1,
                    end_state: state,
                });
            }
        }

        TokenizerExecResult {
            end_state: (!self.is_end(state)).then_some(state),
            matches,
        }
    }

    pub fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {
        let into_matches = |matches: FxHashMap<_, _>| {
            matches
                .into_iter()
                .map(|(id, (width, end_state))| TokenizerMatch { id, width, end_state })
                .collect()
        };

        let mut state = start;
        let mut matches = FxHashMap::<TerminalID, (usize, u32)>::default();

        for (index, &byte) in input.iter().enumerate() {
            let Some(next) = self.step(state, byte) else {
                return TokenizerExecResult {
                    end_state: None,
                    matches: into_matches(matches),
                };
            };
            state = next;

            for terminal in self.dfa.finalizers(state).iter() {
                matches.insert(terminal as TerminalID, (index + 1, state));
            }
        }

        TokenizerExecResult {
            end_state: Some(state),
            matches: into_matches(matches),
        }
    }

    pub fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
        let exec = self.execute_from_state_all_widths(input, start);
        let end_state = exec.end_state.unwrap_or(start);
        let mut grouped = std::collections::BTreeMap::<usize, BTreeSet<TerminalID>>::new();
        for matched in exec.matches {
            grouped.entry(matched.width).or_default().insert(matched.id);
        }
        TokenizerResult {
            end_state,
            matches: grouped.into_iter().collect(),
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
}
