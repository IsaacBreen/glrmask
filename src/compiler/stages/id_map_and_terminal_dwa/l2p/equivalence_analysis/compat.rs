//! Flattened tokenizer-DFA views for the equivalence-analysis passes.

#[cfg(test)]
use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;

fn build_transition_table(
    transitions: impl Iterator<Item = (u8, u32)>,
) -> [u32; 256] {
    let mut table = [u32::MAX; 256];
    for (byte, target) in transitions {
        table[byte as usize] = target;
    }
    table
}

fn collect_group_ids(groups: impl Iterator<Item = usize>) -> Vec<usize> {
    groups.collect()
}

/// Pre-extracted DFA state data in analysis-compatible format.
#[derive(Debug, Clone)]
pub struct FlatDfaState {
    /// Transition table: `transitions[byte]` = target state index, or `usize::MAX` for no transition.
    pub transitions: [u32; 256],
    /// Sorted list of group IDs that finalize at this state.
    pub finalizers: Vec<usize>,
    /// Sorted list of group IDs reachable from this state.
    pub possible_future_group_ids: Vec<usize>,
}

/// Pre-extracted DFA in the format used by equivalence analysis.
/// The analysis passes only depend on this struct, not on live glrmask types.
#[derive(Debug, Clone)]
pub struct FlatDfa {
    pub states: Vec<FlatDfaState>,
    pub start_state: usize,
}

pub(crate) fn compute_byte_classes(dfa: &FlatDfa) -> [u8; 256] {
    // Hash each byte's transition column using row-major access for cache efficiency.
    // This avoids allocating 256 Vec<u32> of N elements each (was ~35MB for N=34888).
    let mut column_hashes = [0u64; 256];
    for state in &dfa.states {
        for (b, &target) in state.transitions.iter().enumerate() {
            column_hashes[b] = column_hashes[b]
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(target as u64);
        }
    }

    // Sort bytes by hash to find equivalence classes.
    let mut sorted_indices: [u8; 256] = std::array::from_fn(|i| i as u8);
    sorted_indices.sort_unstable_by_key(|&b| column_hashes[b as usize]);

    let mut byte_to_class = [0u8; 256];
    let mut next_class = 0u8;
    byte_to_class[sorted_indices[0] as usize] = 0;

    for i in 1..256 {
        let curr = sorted_indices[i];
        let h = column_hashes[curr as usize];

        if h != column_hashes[sorted_indices[i - 1] as usize] {
            // Different hash → different class.
            next_class += 1;
            byte_to_class[curr as usize] = next_class;
        } else {
            // Same hash → verify by comparing transition columns.
            // Check all prior bytes with the same hash (handles rare collisions).
            let mut assigned = false;
            for j in (0..i).rev() {
                let prev = sorted_indices[j];
                if column_hashes[prev as usize] != h {
                    break;
                }
                let same = dfa.states.iter().all(|state| {
                    state.transitions[curr as usize] == state.transitions[prev as usize]
                });
                if same {
                    byte_to_class[curr as usize] = byte_to_class[prev as usize];
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                next_class += 1;
                byte_to_class[curr as usize] = next_class;
            }
        }
    }

    byte_to_class
}

impl FlatDfa {
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let dfa = &tokenizer.dfa;
        let dfa_states = dfa.states();
        let start_state = tokenizer.start_state() as usize;
        let states: Vec<FlatDfaState> = dfa_states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let transitions = build_transition_table(
                    state.transitions.iter().map(|(byte, &target)| (byte, target)),
                );
                let finalizers = collect_group_ids(state.finalizers.iter());
                let possible_future_group_ids =
                    collect_group_ids(dfa.possible_future_group_ids(i as u32).iter());

                FlatDfaState {
                    transitions,
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

    /// Build a FlatDfa filtering finalizers and futures to only active groups.
    /// States that differ only by inactive-group data become equivalent.
    pub fn from_tokenizer_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {
        let dfa = &tokenizer.dfa;
        let dfa_states = dfa.states();
        let start_state = tokenizer.start_state() as usize;
        let num_groups = active_groups.len();
        let states: Vec<FlatDfaState> = dfa_states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let transitions = build_transition_table(
                    state.transitions.iter().map(|(byte, &target)| (byte, target)),
                );
                let finalizers: Vec<usize> = state.finalizers.iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();
                let possible_future_group_ids: Vec<usize> = dfa
                    .possible_future_group_ids(i as u32)
                    .iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();

                FlatDfaState {
                    transitions,
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

    /// Build a FlatDfa using a pre-built flat transition table (state-major layout:
    /// `flat_trans[state * 256 + byte] = target`), filtering finalizers/futures
    /// to only active groups. Avoids re-iterating CharTransitions per state.
    pub fn from_flat_trans_filtered(
        flat_trans: &[u32],
        tokenizer: &Tokenizer,
        active_groups: &[bool],
    ) -> Self {
        let dfa = &tokenizer.dfa;
        let dfa_states = dfa.states();
        let start_state = tokenizer.start_state() as usize;
        let num_groups = active_groups.len();
        let states: Vec<FlatDfaState> = dfa_states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let base = i * 256;
                let mut transitions = [u32::MAX; 256];
                transitions.copy_from_slice(&flat_trans[base..base + 256]);
                let finalizers: Vec<usize> = state.finalizers.iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();
                let possible_future_group_ids: Vec<usize> = dfa
                    .possible_future_group_ids(i as u32)
                    .iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();
                FlatDfaState {
                    transitions,
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();
        FlatDfa { states, start_state }
    }

    #[cfg(test)]
    pub fn num_states(&self) -> usize {
        self.states.len()
    }
}

/// A thin wrapper around glrmask's `Tokenizer` that exposes the flattened DFA.
///
/// The equivalence-analysis code calls `dfa()` and accesses `.states` and
/// `.start_state` directly.
/// This wrapper pre-extracts all data into `FlatDfa` on construction.
pub struct TokenizerView {
    pub flat_dfa: FlatDfa,
}

#[cfg(test)]
fn execution_result(
    match_positions: &BTreeMap<usize, usize>,
    end_state: Option<usize>,
) -> ExecuteResult {
    ExecuteResult {
        matches: collect_matches(match_positions),
        end_state,
    }
}

impl TokenizerView {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_tokenizer(tokenizer),
        }
    }

    /// Build a view that filters finalizers and futures to only active groups.
    pub fn new_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_tokenizer_filtered(tokenizer, active_groups),
        }
    }

    /// Build a filtered view using a pre-built flat transition table.
    /// Avoids re-iterating CharTransitions — just copies from flat_trans.
    pub fn new_filtered_from_flat_trans(
        flat_trans: &[u32],
        tokenizer: &Tokenizer,
        active_groups: &[bool],
    ) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_flat_trans_filtered(flat_trans, tokenizer, active_groups),
        }
    }

    pub fn dfa(&self) -> &FlatDfa {
        &self.flat_dfa
    }

    pub fn initial_state_id(&self) -> usize {
        self.flat_dfa.start_state
    }

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
                return execution_result(&match_positions, None);
            }

            current = next as usize;
            let position = pos + 1;
            for &gid in &dfa.states[current].finalizers {
                match_positions.insert(gid, position);
            }

            if dfa.states[current].possible_future_group_ids.is_empty() {
                return execution_result(&match_positions, None);
            }
        }

        execution_result(
            &match_positions,
            (!state_is_done(&dfa.states[current])).then_some(current),
        )
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
        let tokenizer_view = TokenizerView::new(&tokenizer);

        let result = tokenizer_view.execute_from_state_nonzero(b"11", tokenizer_view.initial_state_id());

        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].group_id, 0);
        assert_eq!(result.matches[0].position, 2);
    }

    #[test]
    fn test_execute_from_state_nonzero_returns_none_for_sink_end_state() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a")]);
        let tokenizer_view = TokenizerView::new(&tokenizer);

        let result = tokenizer_view.execute_from_state_nonzero(b"a", tokenizer_view.initial_state_id());

        assert_eq!(result.end_state, None);
    }
}
