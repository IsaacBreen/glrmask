//! Flattened tokenizer-DFA views for the equivalence-analysis passes.

#[cfg(test)]
use std::collections::BTreeMap;

use std::sync::Arc;

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

/// Per-state metadata: finalizers and reachable groups.
/// Transitions are stored separately in `FlatDfa::transitions` for sharing.
#[derive(Debug, Clone)]
pub struct FlatDfaState {
    /// Sorted list of group IDs that finalize at this state.
    pub finalizers: Vec<usize>,
    /// Sorted list of group IDs reachable from this state.
    pub possible_future_group_ids: Vec<usize>,
}

/// Pre-extracted DFA in the format used by equivalence analysis.
/// Transitions are stored contiguously in a flat table (`transitions[state * 256 + byte]`),
/// separated from per-state metadata to enable zero-copy sharing across partitions via `Arc`.
#[derive(Debug, Clone)]
pub struct FlatDfa {
    pub states: Vec<FlatDfaState>,
    pub start_state: usize,
    /// Flat transition table: `transitions[state * 256 + byte] = target_state`.
    /// Shared via `Arc` to avoid 35MB duplication per partition.
    pub transitions: Arc<[u32]>,
}

pub(crate) fn compute_byte_classes(dfa: &FlatDfa) -> [u8; 256] {
    // Hash each byte's transition column using row-major access for cache efficiency.
    let mut column_hashes = [0u64; 256];
    let num_states = dfa.states.len();
    for s in 0..num_states {
        let base = s * 256;
        for b in 0..256 {
            column_hashes[b] = column_hashes[b]
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(dfa.transitions[base + b] as u64);
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
            next_class += 1;
            byte_to_class[curr as usize] = next_class;
        } else {
            let mut assigned = false;
            for j in (0..i).rev() {
                let prev = sorted_indices[j];
                if column_hashes[prev as usize] != h {
                    break;
                }
                let same = (0..num_states).all(|s| {
                    let base = s * 256;
                    dfa.transitions[base + curr as usize] == dfa.transitions[base + prev as usize]
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
    /// Get the transition target for a given state and byte.
    #[inline]
    pub fn trans(&self, state: usize, byte: usize) -> u32 {
        self.transitions[state * 256 + byte]
    }

    /// Get the 256-entry transition slice for a given state.
    #[inline]
    pub fn transitions_for(&self, state: usize) -> &[u32] {
        let base = state * 256;
        &self.transitions[base..base + 256]
    }
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let dfa = &tokenizer.dfa;
        let dfa_states = dfa.states();
        let start_state = tokenizer.start_state() as usize;
        let num_states = dfa_states.len();
        let mut transitions = vec![u32::MAX; num_states * 256];
        let states: Vec<FlatDfaState> = dfa_states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let base = i * 256;
                for (byte, &target) in state.transitions.iter() {
                    transitions[base + byte as usize] = target;
                }
                let finalizers = collect_group_ids(state.finalizers.iter());
                let possible_future_group_ids =
                    collect_group_ids(dfa.possible_future_group_ids(i as u32).iter());

                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();

        FlatDfa {
            states,
            start_state,
            transitions: Arc::from(transitions),
        }
    }

    /// Build a FlatDfa filtering finalizers and futures to only active groups.
    pub fn from_tokenizer_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {
        let dfa = &tokenizer.dfa;
        let dfa_states = dfa.states();
        let start_state = tokenizer.start_state() as usize;
        let num_groups = active_groups.len();
        let num_states = dfa_states.len();
        let mut transitions = vec![u32::MAX; num_states * 256];
        let states: Vec<FlatDfaState> = dfa_states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let base = i * 256;
                for (byte, &target) in state.transitions.iter() {
                    transitions[base + byte as usize] = target;
                }
                let finalizers: Vec<usize> = state.finalizers.iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();
                let possible_future_group_ids: Vec<usize> = dfa
                    .possible_future_group_ids(i as u32)
                    .iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();

                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();

        FlatDfa {
            states,
            start_state,
            transitions: Arc::from(transitions),
        }
    }

    /// Build a FlatDfa using a pre-built flat transition table, sharing the
    /// transition data via Arc. Only finalizers/futures are allocated per partition.
    pub fn from_flat_trans_filtered(
        flat_trans: &Arc<[u32]>,
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
                let finalizers: Vec<usize> = state.finalizers.iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();
                let possible_future_group_ids: Vec<usize> = dfa
                    .possible_future_group_ids(i as u32)
                    .iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();
                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();
        FlatDfa { states, start_state, transitions: Arc::clone(flat_trans) }
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

    /// Build a filtered view using a pre-built shared flat transition table.
    /// Shares transition data via Arc — zero-copy per partition.
    pub fn new_filtered_from_flat_trans(
        flat_trans: &Arc<[u32]>,
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
            let next = dfa.trans(current, byte as usize);
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
            (!state_is_done(dfa, current)).then_some(current),
        )
    }
}

#[cfg(test)]
fn state_is_done(dfa: &FlatDfa, state: usize) -> bool {
    dfa.transitions_for(state).iter().all(|&target| target == u32::MAX)
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
