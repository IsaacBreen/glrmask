
pub(crate) mod disallowed_follows;
pub mod combined;
pub mod compat;
pub mod state;
pub mod vocab;
pub mod combined_equivalence_analysis;
pub mod reference;

#[derive(Debug, Clone)]
pub struct ManyToOneIdMap {
    pub original_to_internal: Vec<u32>,
    pub internal_to_originals: Vec<Vec<u32>>,
    pub representative_original_ids: Vec<u32>,
}

impl ManyToOneIdMap {
    /// Construct from a pre-computed original→internal mapping with explicit representatives.
    pub fn from_original_to_internal_with_representatives(
        original_to_internal: Vec<u32>,
        num_internal: u32,
        representative_original_ids: Vec<u32>,
    ) -> Self {
        let mut internal_to_originals = vec![Vec::new(); num_internal as usize];
        for (original, &internal) in original_to_internal.iter().enumerate() {
            if (internal as usize) < internal_to_originals.len() {
                internal_to_originals[internal as usize].push(original as u32);
            }
        }
        Self {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        }
    }

    pub fn num_internal_ids(&self) -> u32 {
        self.internal_to_originals.len() as u32
    }

    pub fn representative_original_id_for_internal(&self, internal_id: u32) -> Option<u32> {
        self.representative_original_ids
            .get(internal_id as usize)
            .copied()
    }

    pub fn iter_representative_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.representative_original_ids.iter().copied()
    }

    #[cfg(test)]
    pub fn original_ids_for_internal(&self, internal_id: u32) -> Option<&Vec<u32>> {
        self.internal_to_originals.get(internal_id as usize)
    }

    pub fn internal_to_originals_vecs(&self) -> Vec<Vec<u32>> {
        self.internal_to_originals.clone()
    }

    #[cfg(test)]
    pub fn max_original_id(&self) -> u32 {
        self.original_to_internal
            .len()
            .checked_sub(1)
            .map(|i| i as u32)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct InternalIdMap {
    pub tokenizer_states: ManyToOneIdMap,
    pub vocab_tokens: ManyToOneIdMap,
}

impl InternalIdMap {
    pub fn build(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        vocab: &crate::Vocab,
        disallowed_follows: &std::collections::BTreeMap<u32, crate::ds::bitset::BitSet>,
        ignore_terminal: Option<u32>,
    ) -> Self {
        combined::analyze_equivalences(tokenizer, vocab, disallowed_follows, ignore_terminal)
    }

    /// L1 fast path: compute equivalence classes using direct DFA fingerprinting.
    /// Only valid when all terminals have max path length ≤ 1.
    pub fn build_l1(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        vocab: &crate::Vocab,
    ) -> Self {
        combined::analyze_equivalences_l1(tokenizer, vocab)
    }

    /// Build a trivial identity map where each tokenizer state and vocab token
    /// maps to its own singleton equivalence class (no merging).
    #[cfg(test)]
    pub fn build_identity(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        vocab: &crate::Vocab,
    ) -> Self {
        let num_states = tokenizer.num_states() as usize;
        let tokenizer_states = ManyToOneIdMap {
            original_to_internal: (0..num_states as u32).collect(),
            internal_to_originals: (0..num_states as u32)
                .map(|i| vec![i])
                .collect(),
            representative_original_ids: (0..num_states as u32).collect(),
        };

        let max_token_id = vocab.entries.keys().last().copied().unwrap_or(0) as usize;
        let mut original_to_internal = vec![u32::MAX; max_token_id + 1];
        let mut internal_to_originals = Vec::new();
        let mut representative_original_ids = Vec::new();
        for &token_id in vocab.entries.keys() {
            let internal_id = internal_to_originals.len() as u32;
            original_to_internal[token_id as usize] = internal_id;
            internal_to_originals.push(vec![token_id]);
            representative_original_ids.push(token_id);
        }
        let vocab_tokens = ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        };

        Self {
            tokenizer_states,
            vocab_tokens,
        }
    }

    pub fn num_tsids(&self) -> u32 {
        self.tokenizer_states.num_internal_ids()
    }

    pub fn num_internal_tokens(&self) -> u32 {
        self.vocab_tokens.num_internal_ids()
    }

    pub fn max_internal_token_id(&self) -> u32 {
        self.num_internal_tokens().saturating_sub(1)
    }

    #[cfg(test)]
    pub fn max_token_id(&self) -> u32 {
        self.vocab_tokens.max_original_id()
    }
}

/// Compute a refined global `InternalIdMap` from partition-specific id maps.
///
/// Tokenizer state equivalence is the **finest common refinement**: two states
/// are globally equivalent iff they are equivalent in every partition.
///
/// Token equivalence classes from disjoint partitions are concatenated.
pub fn refine_partition_id_maps(
    partition_maps: &[InternalIdMap],
    num_tokenizer_states: usize,
    full_vocab_max_token_id: u32,
) -> InternalIdMap {
    use std::collections::HashMap;

    // --- State refinement: composite key = (class_P0, class_P1, ...) ---
    let mut composite_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut state_original_to_internal = vec![0u32; num_tokenizer_states];
    let mut state_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut state_representative_ids: Vec<u32> = Vec::new();

    for state in 0..num_tokenizer_states {
        let composite: Vec<u32> = partition_maps
            .iter()
            .map(|m| m.tokenizer_states.original_to_internal[state])
            .collect();
        let next_id = state_internal_to_originals.len() as u32;
        let class = *composite_to_class.entry(composite).or_insert_with(|| {
            state_internal_to_originals.push(Vec::new());
            state_representative_ids.push(state as u32);
            next_id
        });
        state_original_to_internal[state] = class;
        state_internal_to_originals[class as usize].push(state as u32);
    }

    // --- Token refinement: concatenate partition classes ---
    let mut token_original_to_internal = vec![u32::MAX; full_vocab_max_token_id as usize + 1];
    let mut token_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut token_representative_ids: Vec<u32> = Vec::new();

    for id_map in partition_maps {
        let offset = token_internal_to_originals.len() as u32;
        for (partition_internal, originals) in id_map.vocab_tokens.internal_to_originals.iter().enumerate() {
            token_internal_to_originals.push(originals.clone());
            token_representative_ids.push(
                id_map.vocab_tokens.representative_original_ids[partition_internal],
            );
            for &orig in originals {
                token_original_to_internal[orig as usize] = offset + partition_internal as u32;
            }
        }
    }

    InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_original_to_internal,
            internal_to_originals: state_internal_to_originals,
            representative_original_ids: state_representative_ids,
        },
        vocab_tokens: ManyToOneIdMap {
            original_to_internal: token_original_to_internal,
            internal_to_originals: token_internal_to_originals,
            representative_original_ids: token_representative_ids,
        },
    }
}
