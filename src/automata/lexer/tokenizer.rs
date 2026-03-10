//! NOTE: regex parsing and compilation helpers live in `regex.rs`.
//! Keep this file focused on the runtime-facing tokenizer surface.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::automata::dfa::DFA;
use crate::compiler::grammar_def::TerminalID;
use crate::ds::u8set::U8Set;

pub use super::regex::parse_regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub dfa: DFA,
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
    pub fn drain_nullable_terminals(&mut self) -> BTreeSet<TerminalID> {
        let nullable = self.matched_terminals(self.start_state());
        for &tid in &nullable {
            self.dfa.clear_finalizer(self.start_state(), tid);
        }
        nullable
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
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

    pub fn all_matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.matched_terminals(state)
    }

    pub fn possible_future_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.dfa
            .possible_future_group_ids(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
            .collect()
    }

    pub fn terminal_matches(&self, state: u32, terminal: TerminalID) -> bool {
        self.all_matched_terminals(state).contains(&terminal)
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
                    end_state: state,
                });
            }
        }

        TokenizerExecResult {
            end_state: Some(state),
            matches,
        }
    }

    pub fn execute(&self, input: &[u8], start: u32) -> (u32, BTreeSet<TerminalID>) {
        let mut state = start;
        for &byte in input {
            let Some(next) = self.step(state, byte) else {
                return (state, BTreeSet::new());
            };
            state = next;
        }
        (state, self.all_matched_terminals(state))
    }

    pub fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
        let exec = self.execute_from_state(input, start);
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

    pub fn tokens_accessible_from_state(&self, state: u32) -> BTreeSet<TerminalID> {
        self.possible_future_terminals(state)
    }

    pub fn execute_all_matches_cb<F>(&self, input: &[u8], start: u32, cb: F) -> u32
    where
        F: FnMut(usize, &BTreeSet<u32>),
    {
        let result = self.execute_all_matches(input, start);
        let mut cb = cb;
        for (offset, matches) in &result.matches {
            let mapped: BTreeSet<u32> = matches.iter().copied().collect();
            cb(*offset, &mapped);
        }
        result.end_state
    }

    pub fn execute_all_matches_cb_filtered<F>(
        &self,
        input: &[u8],
        start: u32,
        state_has_used: &[bool],
        cb: F,
    ) -> u32
    where
        F: FnMut(usize, &BTreeSet<u32>),
    {
        let _ = state_has_used;
        self.execute_all_matches_cb(input, start, cb)
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
