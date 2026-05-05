use super::compat::TokenizerView;
use super::vocab::fast::VocabEquivalenceResult;

pub(crate) struct TokenDedup<'a> {
    pub(crate) representative_token_bytes: Vec<&'a [u8]>,
    pub(crate) original_to_repr: Vec<usize>,
}

#[inline]
pub(crate) fn hash_byte_class_seq(bytes: &[u8], byte_to_class: &[u8; 256]) -> u128 {
    let mut hash: u128 = 0xFF51_AFD7_ED55_8CCD;
    hash = hash.wrapping_mul(0xC4CE_B9FE_1A85_EC53).wrapping_add(bytes.len() as u128);
    for &byte in bytes {
        hash = hash
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(byte_to_class[byte as usize] as u128);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    hash ^= hash >> 29;
    hash
}

pub(crate) fn expand_vocab_classes(
    dedup_classes: VocabEquivalenceResult,
    original_to_repr: &[usize],
    num_representatives: usize,
) -> VocabEquivalenceResult {
    let mut repr_to_class = vec![usize::MAX; num_representatives];
    let mut original_classes: Vec<Vec<usize>> = Vec::with_capacity(dedup_classes.len());

    for (class_idx, dedup_class) in dedup_classes.iter().enumerate() {
        for &repr_idx in dedup_class {
            repr_to_class[repr_idx] = class_idx;
        }
        original_classes.push(Vec::new());
    }

    for (original_idx, &repr_idx) in original_to_repr.iter().enumerate() {
        let class_idx = repr_to_class[repr_idx];
        debug_assert!(class_idx != usize::MAX);
        original_classes[class_idx].push(original_idx);
    }

    original_classes
        .into_iter()
        .filter(|class| !class.is_empty())
        .collect()
}

pub(crate) fn representative_tokens_for_vocab_classes<'a>(
    dedup_vocab_classes: &VocabEquivalenceResult,
    representative_token_bytes: &'a [&'a [u8]],
) -> Vec<&'a [u8]> {
    dedup_vocab_classes
        .iter()
        .map(|dedup_class| representative_token_bytes[dedup_class[0]])
        .collect()
}

pub(crate) fn tokenizer_group_count(tokenizer: &TokenizerView) -> usize {
    tokenizer
        .dfa()
        .states
        .iter()
        .flat_map(|state| {
            state
                .finalizers
                .iter()
                .copied()
                .chain(state.possible_future_group_ids.iter().copied())
        })
        .max()
        .map_or(0, |max_group| max_group + 1)
}