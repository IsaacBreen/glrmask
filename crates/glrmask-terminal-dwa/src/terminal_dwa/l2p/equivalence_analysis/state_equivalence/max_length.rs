use crate::automata::lexer::Lexer;
use rayon::prelude::*;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::build_state_map_from_subset_representatives;
use super::super::compat::TokenizerView;

#[derive(Debug, Clone, Copy)]
pub enum MaxLengthMode {
    StableByteRestricted,
    KBoundedByteRestricted,
}

#[derive(Debug, Clone)]
pub struct MaxLengthStatistic {
    max_token_len: usize,
    relevant_bytes: [bool; 256],
}

impl crate::vocab::VocabDerivedArtifact for MaxLengthStatistic {}

impl MaxLengthStatistic {
    pub fn max_token_len(&self) -> usize {
        self.max_token_len
    }

    pub fn relevant_byte_count(&self) -> usize {
        self.relevant_bytes.iter().filter(|&&active| active).count()
    }

    pub fn relevant_bytes(&self) -> &[bool; 256] {
        &self.relevant_bytes
    }
}

fn is_active_group(group_id: usize, active_groups: Option<&[bool]>) -> bool {
    active_groups.is_none_or(|groups| {
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
        let starts_new_label = previous_state.is_none_or(|prev| {
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
    relevant_bytes.is_none_or(|bytes| bytes[byte])
}

pub fn active_byte_representatives(
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

fn stable_representatives(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    active_bytes: &[u8],
    states: &[usize],
) -> Vec<usize> {
    let (initial, count) = build_initial_label_partition(tokenizer, active_groups);
    if count == initial.len() { return states.to_vec(); }
    let (classes, representatives, _) = crate::terminal_dwa::congruence::refine_partition_rows(
        initial,
        |state| active_bytes.iter().filter_map(move |&byte| {
            let target = tokenizer.get_transition(state as u32, byte);
            (target != u32::MAX).then_some((byte, target))
        }),
        None,
    );
    states.iter().map(|&s| representatives[classes[s] as usize] as usize).collect()
}

pub fn compute_statistic(vocab: &Vocab) -> MaxLengthStatistic {
    let mut relevant_bytes = [false; 256];
    let mut max_token_len = 0usize;
    for bytes in vocab.entries_map().values() {
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
pub fn cached_statistic(vocab: &Vocab) -> std::sync::Arc<MaxLengthStatistic> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<MaxLengthStatistic>() {
        return cached;
    }
    let statistic = std::sync::Arc::new(compute_statistic(vocab));
    vocab.vocab_derived_cache_set(std::sync::Arc::clone(&statistic));
    statistic
}

pub fn compute_state_map(
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
            stable_representatives(tokenizer, active_groups, &active_bytes, &states)
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
