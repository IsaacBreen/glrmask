//! NOTE: regex parsing and compilation helpers live in `tokenizer_regex.rs`.
//! Keep this file focused on the runtime-facing tokenizer surface.
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

impl Tokenizer {
    pub fn start_state(&self) -> u32 {
        unimplemented!()
    }

    pub fn step(&self, _state: u32, _byte: u8) -> Option<u32> {
        unimplemented!()
    }

    pub fn run(&self, _input: &[u8]) -> u32 {
        unimplemented!()
    }

    pub fn matched_terminals(&self, _state: u32) -> BTreeSet<TerminalID> {
        unimplemented!()
    }

    pub fn matched_non_greedy_terminals(&self, _state: u32) -> BTreeSet<TerminalID> {
        unimplemented!()
    }

    pub fn possible_future_terminals(&self, _state: u32) -> BTreeSet<TerminalID> {
        unimplemented!()
    }

    pub fn terminal_matches(&self, _state: u32, _terminal: TerminalID) -> bool {
        unimplemented!()
    }

    pub fn num_states(&self) -> u32 {
        unimplemented!()
    }

    pub fn compute_reachable_terminals(&self) -> Vec<BTreeSet<TerminalID>> {
        unimplemented!()
    }

    pub fn execute(&self, _input: &[u8], _start: u32) -> (u32, BTreeSet<TerminalID>) {
        unimplemented!()
    }

    pub fn execute_all_matches(&self, _input: &[u8], _start: u32) -> TokenizerResult {
        unimplemented!()
    }

    pub fn initial_state(&self) -> u32 {
        unimplemented!()
    }

    pub fn execute_all_matches_cb<F>(&self, _input: &[u8], _start: u32, _cb: F) -> u32
    where
        F: FnMut(usize, &BTreeSet<u32>),
    {
        unimplemented!()
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
        unimplemented!()
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
