use super::compat::FlatDfa;
use super::vocab::fast::VocabEquivalenceResult;

pub struct TokenDedup<'a> {
    pub representative_token_bytes: Vec<&'a [u8]>,
    /// Input-token position that supplied each representative byte string.
    /// This is pure provenance; it does not participate in the equivalence
    /// relation, but lets vocabulary-derived artifacts address the same token
    /// without byte-string lookups.
    pub representative_original_indices: Vec<usize>,
    pub original_to_repr: Vec<usize>,
}

#[inline]
pub fn hash_byte_class_seq(bytes: &[u8], byte_to_class: &[u8; 256]) -> u128 {
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

pub fn representative_tokens_for_vocab_classes<'a>(
    dedup_vocab_classes: &VocabEquivalenceResult,
    representative_token_bytes: &'a [&'a [u8]],
) -> Vec<&'a [u8]> {
    dedup_vocab_classes
        .iter()
        .map(|dedup_class| representative_token_bytes[dedup_class[0]])
        .collect()
}

pub fn tokenizer_group_count(tokenizer: &FlatDfa) -> usize {
    tokenizer
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
