//! Compatibility layer bridging glrmask's DFA types to the interface expected by
//! the ported sep1 equivalence analysis code.
//!
//! The sep1 code expects:
//! - `Tokenizer` with `.dfa()` returning a DFA struct that has public
//!   `.states`, `.start_state`
//! - `DFAState` with public `.transitions: CharTransitions<usize>`,
//!   `.finalizers: DenseStateSet`, `.possible_future_group_ids: BTreeSet<GroupID>`
//! - State indices as `usize`
//!
//! glrmask uses:
//! - `Tokenizer` with public `.dfa` field
//! - `DFAState` with `BitSet` finalizers, private `possible_future_group_ids`
//! - State indices as `u32`
//!
//! This module provides flat, pre-extracted views of the DFA data so the sep1
//! code can be adapted with minimal changes.


#[cfg(test)]
use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;

/// Pre-extracted DFA state data in sep1-compatible format.
#[derive(Debug, Clone)]
pub struct FlatDfaState {
    /// Transition table: `transitions[byte]` = target state index, or `usize::MAX` for no transition.
    pub transitions: [u32; 256],
    /// Sorted list of group IDs that finalize at this state.
    pub finalizers: Vec<usize>,
    /// Sorted list of group IDs reachable from this state.
    pub possible_future_group_ids: Vec<usize>,
}

/// Pre-extracted DFA in sep1-compatible format.
/// All the sep1 code needs is this struct; no reference to glrmask types at runtime.
#[derive(Debug, Clone)]
pub struct FlatDfa {
    pub states: Vec<FlatDfaState>,
    pub start_state: usize,
}

impl FlatDfa {
    /// Extract a flat DFA from a glrmask Tokenizer.
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let dfa = &tokenizer.dfa;
        let dfa_states = dfa.states();
        let start_state = tokenizer.start_state() as usize;
        let states: Vec<FlatDfaState> = dfa_states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let mut table = [u32::MAX; 256];
                for (byte, &target) in state.transitions.iter() {
                    table[byte as usize] = target;
                }

                let finalizers: Vec<usize> = state.finalizers.iter().collect();
                let possible_future_group_ids: Vec<usize> =
                    dfa.possible_future_group_ids(i as u32).iter().collect();

                FlatDfaState {
                    transitions: table,
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();

        FlatDfa {
            states,
            start_state,
        }
    }

    #[cfg(test)]
    pub fn num_states(&self) -> usize {
        self.states.len()
    }
}

/// A thin wrapper around glrmask's `Tokenizer` that provides sep1-compatible accessors.
///
/// The sep1 code calls `regex.dfa()` and accesses `.states`, `.start_state`, etc.
/// This wrapper pre-extracts all data into `FlatDfa` on construction.
pub struct Sep1Tokenizer {
    pub flat_dfa: FlatDfa,
}

impl Sep1Tokenizer {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        Sep1Tokenizer {
            flat_dfa: FlatDfa::from_tokenizer(tokenizer),
        }
    }

    /// Returns the flat DFA (analogous to sep1's `Tokenizer::dfa()`).
    pub fn dfa(&self) -> &FlatDfa {
        &self.flat_dfa
    }

    /// Start state (analogous to sep1's `Tokenizer::initial_state_id().0`).
    pub fn initial_state_id(&self) -> usize {
        self.flat_dfa.start_state
    }

    /// Run the DFA from a given state on input bytes using sep1-compatible execution semantics.
    #[cfg(test)]
    pub fn execute_from_state_nonzero(&self, input: &[u8], start_state: usize) -> ExecuteResult {
        let dfa = &self.flat_dfa;
        if start_state >= dfa.states.len() {
            return ExecuteResult {
                matches: Vec::new(),
                end_state: None,
            };
        }

        let mut current = start_state;
        let mut match_positions: BTreeMap<usize, usize> = dfa.states[current]
            .finalizers
            .iter()
            .copied()
            .map(|group_id| (group_id, 0usize))
            .collect();

        for (pos, &byte) in input.iter().enumerate() {
            let next = dfa.states[current].transitions[byte as usize];
            if next == u32::MAX {
                return ExecuteResult {
                    matches: collect_matches(&match_positions),
                    end_state: None,
                };
            }

            current = next as usize;
            let position = pos + 1;
            for &gid in &dfa.states[current].finalizers {
                match_positions.insert(gid, position);
            }

            if dfa.states[current].possible_future_group_ids.is_empty() {
                return ExecuteResult {
                    matches: collect_matches(&match_positions),
                    end_state: None,
                };
            }
        }

        ExecuteResult {
            matches: collect_matches(&match_positions),
            end_state: (!state_is_done(&dfa.states[current])).then_some(current),
        }
    }
}

#[cfg(test)]
fn state_is_done(state: &FlatDfaState) -> bool {
    state.transitions.iter().all(|&target| target == u32::MAX)
}

#[cfg(test)]
fn collect_matches(match_positions: &BTreeMap<usize, usize>) -> Vec<ExecuteMatch> {
    match_positions
        .iter()
        .filter_map(|(&group_id, &position)| {
            (position != 0).then_some(ExecuteMatch { group_id, position })
        })
        .collect()
}

/// Result of executing the DFA on input bytes.
#[cfg(test)]
pub struct ExecuteResult {
    pub matches: Vec<ExecuteMatch>,
    pub end_state: Option<usize>,
}

/// A single match: group ID and byte position.
#[cfg(test)]
pub struct ExecuteMatch {
    pub group_id: usize,
    pub position: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, plus};
    use crate::compiler::compile::build_tokenizer_from_exprs;

    #[test]
    fn test_execute_from_state_nonzero_deduplicates_group_matches() {
        let tokenizer = build_tokenizer_from_exprs(&[plus(bytes(b"1"))]);
        let sep1 = Sep1Tokenizer::new(&tokenizer);

        let result = sep1.execute_from_state_nonzero(b"11", sep1.initial_state_id());

        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].group_id, 0);
        assert_eq!(result.matches[0].position, 2);
    }

    #[test]
    fn test_execute_from_state_nonzero_returns_none_for_sink_end_state() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a")]);
        let sep1 = Sep1Tokenizer::new(&tokenizer);

        let result = sep1.execute_from_state_nonzero(b"a", sep1.initial_state_id());

        assert_eq!(result.end_state, None);
    }
}
