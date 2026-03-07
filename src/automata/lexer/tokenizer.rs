//! NOTE: regex parsing and compilation helpers live in `tokenizer_regex.rs`.
//! Keep this file focused on the runtime-facing tokenizer surface.
// SEP1_MAP: This file maps most closely to sep1's `dfa_u8/tokenizer_ops.rs`, with the same tokenizer-stepping role split away from regex construction.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::automata::dfa::DFA;
use crate::compiler::grammar_def::TerminalID;

pub use super::tokenizer_regex::parse_regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub dfa: DFA,
    pub num_terminals: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMatch {
    pub id: TerminalID,
    pub width: usize,
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

    pub fn step(&self, _state: u32, _byte: u8) -> Option<u32> {
        self.dfa.step(_state, _byte)
    }

    pub fn run(&self, _input: &[u8]) -> u32 {
        _input
            .iter()
            .try_fold(self.start_state(), |state, &byte| self.step(state, byte))
            .unwrap_or(self.start_state())
    }

    pub fn matched_terminals(&self, _state: u32) -> BTreeSet<TerminalID> {
        self.dfa
            .finalizers(_state)
            .iter()
            .map(|terminal| terminal as TerminalID)
            .collect()
    }

    pub fn matched_non_greedy_terminals(&self, _state: u32) -> BTreeSet<TerminalID> {
        self.dfa
            .non_greedy_finalizers(_state)
            .iter()
            .map(|terminal| terminal as TerminalID)
            .collect()
    }

    pub fn all_matched_terminals(&self, _state: u32) -> BTreeSet<TerminalID> {
        let mut terminals = self.matched_terminals(_state);
        terminals.extend(self.matched_non_greedy_terminals(_state));
        terminals
    }

    pub fn possible_future_terminals(&self, _state: u32) -> BTreeSet<TerminalID> {
        self.dfa
            .possible_future_group_ids(_state)
            .iter()
            .map(|terminal| terminal as TerminalID)
            .collect()
    }

    pub fn terminal_matches(&self, _state: u32, _terminal: TerminalID) -> bool {
        self.all_matched_terminals(_state).contains(&_terminal)
    }

    pub fn num_states(&self) -> u32 {
        self.dfa.num_states() as u32
    }

    pub fn compute_reachable_terminals(&self) -> Vec<BTreeSet<TerminalID>> {
        (0..self.num_states())
            .map(|state| self.possible_future_terminals(state))
            .collect()
    }

    pub fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {
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
            for terminal in self.all_matched_terminals(state) {
                matches.push(TokenizerMatch {
                    id: terminal,
                    width: index + 1,
                });
            }
        }

        TokenizerExecResult {
            end_state: Some(state),
            matches,
        }
    }

    pub fn execute(&self, _input: &[u8], _start: u32) -> (u32, BTreeSet<TerminalID>) {
        let mut state = _start;
        for &byte in _input {
            let Some(next) = self.step(state, byte) else {
                return (state, BTreeSet::new());
            };
            state = next;
        }
        (state, self.all_matched_terminals(state))
    }

    pub fn execute_all_matches(&self, _input: &[u8], _start: u32) -> TokenizerResult {
        let exec = self.execute_from_state(_input, _start);
        let end_state = exec.end_state.unwrap_or(_start);
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

    pub fn tokens_accessible_from_state(&self, state: u32) -> BTreeSet<TerminalID> {
        self.possible_future_terminals(state)
    }

    pub fn execute_all_matches_cb<F>(&self, _input: &[u8], _start: u32, _cb: F) -> u32
    where
        F: FnMut(usize, &BTreeSet<u32>),
    {
        let result = self.execute_all_matches(_input, _start);
        let mut cb = _cb;
        for (offset, matches) in &result.matches {
            let mapped: BTreeSet<u32> = matches.iter().copied().collect();
            cb(*offset, &mapped);
        }
        result.end_state
    }

    pub fn execute_all_matches_cb_filtered<F>(
        &self,
        _input: &[u8],
        _start: u32,
        _state_has_used: &[bool],
        _cb: F,
    ) -> u32
    where
        F: FnMut(usize, &BTreeSet<u32>),
    {
        let _ = _state_has_used;
        self.execute_all_matches_cb(_input, _start, _cb)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    pub end_state: u32,
    pub matches: Vec<(usize, BTreeSet<TerminalID>)>,
}

#[cfg(test)]
mod tests {
    // NOTE: the old tokenizer tests are intentionally omitted until the
    // sep1-style lexer automata rewrite lands.
}
