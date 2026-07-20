use crate::automata::lexer::Lexer;
use rayon::prelude::*;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::build_state_map_from_subset_representatives;
use super::super::compat::TokenizerView;

const MISSING_BLOCK: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub(crate) enum MaxLengthMode {
    StableByteRestricted,
    KBoundedByteRestricted,
}

impl MaxLengthMode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            MaxLengthMode::StableByteRestricted => "max_length_stable",
            MaxLengthMode::KBoundedByteRestricted => "max_length_kbounded",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MaxLengthStatistic {
    max_token_len: usize,
    relevant_bytes: [bool; 256],
}

impl crate::vocab::VocabDerivedArtifact for MaxLengthStatistic {}

impl MaxLengthStatistic {
    pub(crate) fn max_token_len(&self) -> usize {
        self.max_token_len
    }

    pub(crate) fn relevant_byte_count(&self) -> usize {
        self.relevant_bytes.iter().filter(|&&active| active).count()
    }

    pub(crate) fn relevant_bytes(&self) -> &[bool; 256] {
        &self.relevant_bytes
    }
}

#[inline(always)]
fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

fn is_active_group(group_id: usize, active_groups: Option<&[bool]>) -> bool {
    active_groups.map_or(true, |groups| {
        groups.get(group_id).copied().unwrap_or(false)
    })
}

fn filtered_terminals(
    values: impl Iterator<Item = u32>,
    active_groups: Option<&[bool]>,
) -> Vec<usize> {
    values
        .filter_map(|value| {
            let value = value as usize;
            is_active_group(value, active_groups).then_some(value)
        })
        .collect()
}

fn build_filtered_finalizer_labels(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
) -> Vec<Vec<usize>> {
    (0..tokenizer.num_states() as usize)
        .into_par_iter()
        .map(|state| filtered_terminals(tokenizer.matched_terminals_iter(state as u32), active_groups))
        .collect()
}

fn build_filtered_possible_future_labels(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
) -> Vec<Vec<usize>> {
    (0..tokenizer.num_states() as usize)
        .into_par_iter()
        .map(|state| {
            filtered_terminals(tokenizer.possible_future_terminals_iter(state as u32), active_groups)
        })
        .collect()
}

fn build_has_any_transition_labels(tokenizer: &Tokenizer) -> Vec<bool> {
    (0..tokenizer.num_states() as usize)
        .into_par_iter()
        .map(|state| tokenizer.transitions_from(state as u32).next().is_some())
        .collect()
}

fn build_initial_label_partition(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
) -> (Vec<u32>, usize) {
    let n = tokenizer.num_states() as usize;
    if n == 0 {
        return (Vec::new(), 0);
    }

    let finalizer_labels = build_filtered_finalizer_labels(tokenizer, active_groups);
    let future_labels = build_filtered_possible_future_labels(tokenizer, active_groups);
    let has_any_transition = build_has_any_transition_labels(tokenizer);

    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_unstable_by(|&left, &right| {
        finalizer_labels[left]
            .cmp(&finalizer_labels[right])
            .then_with(|| future_labels[left].cmp(&future_labels[right]))
            .then_with(|| has_any_transition[left].cmp(&has_any_transition[right]))
            .then_with(|| left.cmp(&right))
    });

    let mut label_ids = vec![0u32; n];
    let mut label_count = 0usize;
    let mut previous_state: Option<usize> = None;

    for state in order {
        let starts_new_label = previous_state.map_or(true, |prev| {
            finalizer_labels[state] != finalizer_labels[prev]
                || future_labels[state] != future_labels[prev]
                || has_any_transition[state] != has_any_transition[prev]
        });
        if starts_new_label {
            label_count += 1;
        }
        label_ids[state] = (label_count - 1) as u32;
        previous_state = Some(state);
    }

    (label_ids, label_count)
}

fn byte_is_relevant(byte: usize, relevant_bytes: Option<&[bool; 256]>) -> bool {
    relevant_bytes.map_or(true, |bytes| bytes[byte])
}

pub(crate) fn active_byte_representatives(
    relevant_bytes: Option<&[bool; 256]>,
    byte_to_class: Option<&[u8; 256]>,
) -> Vec<u8> {
    if let Some(byte_to_class) = byte_to_class {
        let num_classes = *byte_to_class.iter().max().unwrap_or(&0) as usize + 1;
        let mut class_rep: Vec<Option<u8>> = vec![None; num_classes];

        for byte in 0..256usize {
            if !byte_is_relevant(byte, relevant_bytes) {
                continue;
            }
            let class = byte_to_class[byte] as usize;
            if class_rep[class].is_none() {
                class_rep[class] = Some(byte as u8);
            }
        }

        class_rep.into_iter().flatten().collect()
    } else {
        (0..256usize)
            .filter(|&byte| byte_is_relevant(byte, relevant_bytes))
            .map(|byte| byte as u8)
            .collect()
    }
}

fn compute_byte_classes(tokenizer: &Tokenizer) -> [u8; 256] {
    let num_states = tokenizer.num_states() as usize;
    let mut column_hashes = [0u64; 256];

    for state in 0..num_states {
        for byte in 0..256usize {
            column_hashes[byte] = column_hashes[byte]
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(tokenizer.get_transition(state as u32, byte as u8) as u64);
        }
    }

    let mut sorted_indices: [u8; 256] = std::array::from_fn(|i| i as u8);
    sorted_indices.sort_unstable_by_key(|&byte| column_hashes[byte as usize]);

    let mut byte_to_class = [0u8; 256];
    let mut next_class = 0u8;
    byte_to_class[sorted_indices[0] as usize] = 0;

    for i in 1..256 {
        let curr = sorted_indices[i];
        let hash = column_hashes[curr as usize];
        if hash != column_hashes[sorted_indices[i - 1] as usize] {
            next_class += 1;
            byte_to_class[curr as usize] = next_class;
            continue;
        }

        let mut assigned = false;
        for j in (0..i).rev() {
            let prev = sorted_indices[j];
            if column_hashes[prev as usize] != hash {
                break;
            }
            let same = (0..num_states).all(|state| {
                tokenizer.get_transition(state as u32, curr)
                    == tokenizer.get_transition(state as u32, prev)
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

    byte_to_class
}

fn same_partition(left: &[u32], left_count: usize, right: &[u32], right_count: usize) -> bool {
    if left.len() != right.len() || left_count != right_count {
        return false;
    }

    let mut left_to_right = vec![u32::MAX; left_count];
    let mut right_to_left = vec![u32::MAX; right_count];

    for (&l, &r) in left.iter().zip(right.iter()) {
        let li = l as usize;
        let ri = r as usize;
        if li >= left_count || ri >= right_count {
            return false;
        }
        if left_to_right[li] == u32::MAX {
            left_to_right[li] = r;
        } else if left_to_right[li] != r {
            return false;
        }

        if right_to_left[ri] == u32::MAX {
            right_to_left[ri] = l;
        } else if right_to_left[ri] != l {
            return false;
        }
    }

    true
}

fn refine_once_sorted(
    tokenizer: &Tokenizer,
    active_bytes: &[u8],
    label_ids: &[u32],
    prev_blocks: &[u32],
    signatures: &mut [u32],
    row_hashes: &mut [u64],
    order: &mut [usize],
) -> (Vec<u32>, usize) {
    let n = prev_blocks.len();
    let width = 1 + active_bytes.len();
    signatures
        .par_chunks_mut(width)
        .zip(row_hashes.par_iter_mut())
        .enumerate()
        .for_each(|(state, (row, row_hash))| {
            row[0] = label_ids[state];
            for (slot, &byte) in active_bytes.iter().enumerate() {
                let target = tokenizer.get_transition(state as u32, byte);
                row[slot + 1] = if target == u32::MAX {
                    MISSING_BLOCK
                } else {
                    prev_blocks[target as usize]
                };
            }
            *row_hash = row.iter().fold(
                mix_u64((row.len() as u64) ^ 0x9E37_79B9_7F4A_7C15),
                |hash, &value| mix_u64(hash ^ (value as u64).wrapping_add(0xA24B_AED4_963E_E407)),
            );
        });

    order.par_sort_unstable_by(|&left, &right| {
        let hash_cmp = row_hashes[left].cmp(&row_hashes[right]);
        if hash_cmp != std::cmp::Ordering::Equal {
            return hash_cmp;
        }
        let left_start = left * width;
        let right_start = right * width;
        signatures[left_start..left_start + width]
            .cmp(&signatures[right_start..right_start + width])
            .then_with(|| left.cmp(&right))
    });

    let mut next_blocks = vec![0u32; n];
    let mut block_count = 0usize;
    let mut previous_state: Option<usize> = None;

    for &state in order.iter() {
        let starts_new_block = previous_state.map_or(true, |prev| {
            if row_hashes[state] != row_hashes[prev] {
                return true;
            }
            let state_start = state * width;
            let prev_start = prev * width;
            signatures[state_start..state_start + width]
                != signatures[prev_start..prev_start + width]
        });
        if starts_new_block {
            block_count += 1;
        }
        next_blocks[state] = (block_count - 1) as u32;
        previous_state = Some(state);
    }

    (next_blocks, block_count)
}

fn build_full_mapping_from_blocks(blocks: &[u32], num_states: usize) -> Vec<usize> {
    let mut rep_for_block = vec![usize::MAX; blocks
        .iter()
        .copied()
        .max()
        .map_or(0, |value| value as usize + 1)];
    let mut mapping = vec![0usize; num_states];

    for (state, &block) in blocks.iter().enumerate() {
        let block = block as usize;
        if rep_for_block[block] == usize::MAX {
            rep_for_block[block] = state;
        }
        mapping[state] = rep_for_block[block];
    }

    mapping
}

fn build_subset_mapping(states: &[usize], full_mapping: &[usize]) -> Vec<usize> {
    states
        .iter()
        .map(|&state| full_mapping[state])
        .collect()
}

fn compute_kbounded_partition(
    tokenizer: &Tokenizer,
    k: usize,
    active_groups: Option<&[bool]>,
    active_bytes: &[u8],
) -> Vec<u32> {
    let n = tokenizer.num_states() as usize;
    let (label_ids, mut block_count) = build_initial_label_partition(tokenizer, active_groups);
    let mut blocks = label_ids.clone();

    if block_count == n || active_bytes.is_empty() {
        return blocks;
    }

    let width = 1 + active_bytes.len();
    let mut signatures = vec![0u32; n * width];
    let mut row_hashes = vec![0u64; n];
    let mut order: Vec<usize> = (0..n).collect();

    for _ in 0..k {
        let (next_blocks, next_count) = refine_once_sorted(
            tokenizer,
            active_bytes,
            &label_ids,
            &blocks,
            &mut signatures,
            &mut row_hashes,
            &mut order,
        );

        let stable = same_partition(&blocks, block_count, &next_blocks, next_count);
        blocks = next_blocks;
        block_count = next_count;
        if stable || block_count == n {
            break;
        }
    }

    blocks
}

fn stable_refinement_blocks(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    active_bytes: &[u8],
) -> Vec<u32> {
    let n = tokenizer.num_states() as usize;
    let (label_ids, mut block_count) = build_initial_label_partition(tokenizer, active_groups);
    let mut blocks = label_ids.clone();

    if block_count == n || active_bytes.is_empty() {
        return blocks;
    }

    let width = 1 + active_bytes.len();
    let mut signatures = vec![0u32; n * width];
    let mut row_hashes = vec![0u64; n];
    let mut order: Vec<usize> = (0..n).collect();

    loop {
        let (next_blocks, next_count) = refine_once_sorted(
            tokenizer,
            active_bytes,
            &label_ids,
            &blocks,
            &mut signatures,
            &mut row_hashes,
            &mut order,
        );

        let stable = same_partition(&blocks, block_count, &next_blocks, next_count);
        blocks = next_blocks;
        block_count = next_count;
        if stable || block_count == n {
            break;
        }
    }

    blocks
}


/// Refine an already-formed total partition until it is a right congruence on
/// `active_bytes`, while preserving that partition as a lower bound.
pub(crate) fn stable_refinement_from_initial_blocks(
    tokenizer: &Tokenizer,
    active_bytes: &[u8],
    initial_blocks: &[u32],
    initial_block_count: usize,
) -> Vec<u32> {
    let n = tokenizer.num_states() as usize;
    assert_eq!(initial_blocks.len(), n);
    assert!(initial_blocks
        .iter()
        .all(|&block| (block as usize) < initial_block_count));
    if initial_block_count == n || active_bytes.is_empty() {
        return initial_blocks.to_vec();
    }

    let width = 1 + active_bytes.len();
    let mut signatures = vec![0u32; n * width];
    let mut row_hashes = vec![0u64; n];
    let mut order: Vec<usize> = (0..n).collect();
    let mut blocks = initial_blocks.to_vec();
    let mut block_count = initial_block_count;
    loop {
        let (next_blocks, next_count) = refine_once_sorted(
            tokenizer,
            active_bytes,
            initial_blocks,
            &blocks,
            &mut signatures,
            &mut row_hashes,
            &mut order,
        );
        let stable = same_partition(&blocks, block_count, &next_blocks, next_count);
        blocks = next_blocks;
        block_count = next_count;
        if stable || block_count == n {
            return blocks;
        }
    }
}


pub(crate) fn compute_statistic(vocab: &Vocab) -> MaxLengthStatistic {
    let mut relevant_bytes = [false; 256];
    let mut max_token_len = 0usize;
    for bytes in vocab.entries.values() {
        max_token_len = max_token_len.max(bytes.len());
        for &byte in bytes {
            relevant_bytes[byte as usize] = true;
        }
    }
    MaxLengthStatistic {
        max_token_len,
        relevant_bytes,
    }
}

/// Return the vocabulary-only max-length statistic, reusing it across grammar compiles.
/// Without this cache, every state-equivalence lane rescans every token byte.
pub(crate) fn cached_statistic(vocab: &Vocab) -> std::sync::Arc<MaxLengthStatistic> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<MaxLengthStatistic>() {
        return cached;
    }
    let statistic = std::sync::Arc::new(compute_statistic(vocab));
    vocab.vocab_derived_cache_set(std::sync::Arc::clone(&statistic));
    statistic
}

pub(crate) fn compute_state_map(
    tokenizer: &Tokenizer,
    statistic: &MaxLengthStatistic,
    initial_state_map: Option<&ManyToOneIdMap>,
    active_groups: Option<&[bool]>,
    mode: MaxLengthMode,
    kbounded_tokenizer_view: Option<&TokenizerView>,
    kbounded_byte_to_class: Option<&[u8; 256]>,
) -> ManyToOneIdMap {
    if tokenizer.has_epsilon_transitions() {
        let depth = match mode {
            MaxLengthMode::StableByteRestricted => super::nfa::RefinementDepth::Stable,
            MaxLengthMode::KBoundedByteRestricted => {
                super::nfa::RefinementDepth::Bounded(statistic.max_token_len)
            }
        };
        return super::nfa::compute_state_map(
            tokenizer,
            &statistic.relevant_bytes,
            active_groups,
            initial_state_map,
            depth,
        );
    }
    let num_states = tokenizer.num_states() as usize;
    let states: Vec<usize> = match initial_state_map {
        Some(map) => map
            .representative_original_ids
            .iter()
            .map(|&state| state as usize)
            .collect(),
        None => (0..num_states).collect(),
    };
    if states.is_empty() {
        return initial_state_map
            .cloned()
            .unwrap_or_else(|| super::identity_state_map(num_states));
    }

    let owned_byte_to_class;
    let byte_to_class = if let Some(byte_to_class) = kbounded_byte_to_class {
        byte_to_class
    } else {
        owned_byte_to_class = compute_byte_classes(tokenizer);
        &owned_byte_to_class
    };
    let active_bytes =
        active_byte_representatives(Some(&statistic.relevant_bytes), Some(byte_to_class));
    let representative_states = match mode {
        MaxLengthMode::StableByteRestricted => {
            let blocks = stable_refinement_blocks(tokenizer, active_groups, &active_bytes);
            let full_mapping = build_full_mapping_from_blocks(&blocks, num_states);
            build_subset_mapping(&states, &full_mapping)
        }
        MaxLengthMode::KBoundedByteRestricted => {
            let owned_tokenizer_view;
            let tokenizer_view = if let Some(tokenizer_view) = kbounded_tokenizer_view {
                debug_assert_eq!(tokenizer_view.dfa().states.len(), num_states);
                tokenizer_view
            } else {
                owned_tokenizer_view = match active_groups {
                    Some(active_groups) => TokenizerView::new_filtered(tokenizer, active_groups),
                    None => TokenizerView::new(tokenizer),
                };
                &owned_tokenizer_view
            };
            super::super::state::max_length::find_state_equivalence_classes_kbounded(
                tokenizer_view,
                &states,
                statistic.max_token_len,
                active_groups,
                Some(&statistic.relevant_bytes),
                Some(byte_to_class),
                "pipeline_kbounded",
            )
        }
    };

    build_state_map_from_subset_representatives(
        &states,
        &representative_states,
        num_states,
        initial_state_map,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_length_statistic_is_cached_per_vocab() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"longer".to_vec()),
            (2, vec![0xff, b'z']),
        ]);

        assert_eq!(vocab.compiler_cache_entry_count(), 0);
        let first = cached_statistic(&vocab);
        let second = cached_statistic(&vocab);

        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert_eq!(vocab.compiler_cache_entry_count(), 1);
        assert_eq!(first.max_token_len(), 6);
        assert!(first.relevant_bytes()[b'a' as usize]);
        assert!(first.relevant_bytes()[b'z' as usize]);
        assert!(first.relevant_bytes()[0xff]);
        assert!(!first.relevant_bytes()[b'q' as usize]);
    }
}
