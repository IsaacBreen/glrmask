//! Flattened tokenizer-DFA views for the equivalence-analysis passes.

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
    pub(crate) fn from_dfa(dfa: &crate::automata::dfa::DFA) -> Self {
        let num_states = dfa.states().len();
        let mut transitions = vec![u32::MAX; num_states * 256];
        let states: Vec<FlatDfaState> = dfa
            .states()
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let base = i * 256;
                for (byte, &target) in state.transitions.iter() {
                    transitions[base + byte as usize] = target;
                }
                FlatDfaState {
                    finalizers: collect_group_ids(state.finalizers.iter()),
                    possible_future_group_ids: collect_group_ids(
                        dfa.possible_future_group_ids(i as u32).iter(),
                    ),
                }
            })
            .collect();
        FlatDfa {
            states,
            start_state: 0,
            transitions: Arc::from(transitions),
        }
    }

    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        let start_state = tokenizer.virtual_original_state_for_runtime(tokenizer.initial_state()) as usize;
        let num_states = tokenizer.num_states() as usize;
        let mut transitions = vec![u32::MAX; num_states * 256];
        let states: Vec<FlatDfaState> = (0..num_states)
            .map(|i| {
                let base = i * 256;
                for byte in 0..=255u8 {
                    transitions[base + byte as usize] = tokenizer.original_state_transition(i as u32, byte);
                }
                let finalizers = collect_group_ids(tokenizer.original_state_finalizers(i as u32).iter());
                let possible_future_group_ids =
                    collect_group_ids(tokenizer.original_state_possible_futures(i as u32).iter());

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
        let start_state = tokenizer.virtual_original_state_for_runtime(tokenizer.initial_state()) as usize;
        let num_groups = active_groups.len();
        let num_states = tokenizer.num_states() as usize;
        let mut transitions = vec![u32::MAX; num_states * 256];
        let states: Vec<FlatDfaState> = (0..num_states)
            .map(|i| {
                let base = i * 256;
                for byte in 0..=255u8 {
                    transitions[base + byte as usize] = tokenizer.original_state_transition(i as u32, byte);
                }
                let finalizers: Vec<usize> = tokenizer.original_state_finalizers(i as u32).iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();
                let possible_future_group_ids: Vec<usize> = tokenizer
                    .original_state_possible_futures(i as u32)
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
    /// transition data via Arc. Copies all finalizers/futures without filtering.
    pub fn from_flat_trans(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
    ) -> Self {
        let start_state = tokenizer.virtual_original_state_for_runtime(tokenizer.initial_state()) as usize;
        let states: Vec<FlatDfaState> = (0..tokenizer.num_states() as usize)
            .map(|i| {
                let finalizers = collect_group_ids(tokenizer.original_state_finalizers(i as u32).iter());
                let possible_future_group_ids =
                    collect_group_ids(tokenizer.original_state_possible_futures(i as u32).iter());
                FlatDfaState {
                    finalizers,
                    possible_future_group_ids,
                }
            })
            .collect();
        FlatDfa { states, start_state, transitions: Arc::clone(flat_trans) }
    }

    /// Build a FlatDfa using a pre-built flat transition table, sharing the
    /// transition data via Arc. Only finalizers/futures are allocated per partition.
    pub fn from_flat_trans_filtered(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
        active_groups: &[bool],
    ) -> Self {
        let start_state = tokenizer.virtual_original_state_for_runtime(tokenizer.initial_state()) as usize;
        let num_groups = active_groups.len();
        let states: Vec<FlatDfaState> = (0..tokenizer.num_states() as usize)
            .map(|i| {
                let finalizers: Vec<usize> = tokenizer.original_state_finalizers(i as u32).iter()
                    .filter(|&gid| gid < num_groups && active_groups[gid])
                    .collect();
                let possible_future_group_ids: Vec<usize> = tokenizer
                    .original_state_possible_futures(i as u32)
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

}

/// A thin wrapper around glrmask's `Tokenizer` that exposes the flattened DFA.
///
/// The equivalence-analysis code calls `dfa()` and accesses `.states` and
/// `.start_state` directly.
/// This wrapper pre-extracts all data into `FlatDfa` on construction.
pub struct TokenizerView {
    pub flat_dfa: FlatDfa,
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

    /// Build a view using a pre-built shared flat transition table (no group filtering).
    /// Shares transition data via Arc — zero-copy.
    pub fn new_from_flat_trans(
        flat_trans: &Arc<[u32]>,
        tokenizer: &Tokenizer,
    ) -> Self {
        TokenizerView {
            flat_dfa: FlatDfa::from_flat_trans(flat_trans, tokenizer),
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

}
