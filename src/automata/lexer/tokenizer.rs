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
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

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
        let nullable = self.dfa.clear_finalizers_for_state(self.start_state()).iter().map(|terminal| terminal as TerminalID).collect();
        nullable
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
    // NOTE: the old tokenizer tests are intentionally omitted until the
    // sep1-style lexer automata rewrite lands.
}
