
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

    /// Build equivalence classes with optional terminal group filtering.
    ///
    /// When `active_groups` is provided, only the specified terminal groups are
    /// considered during vocab equivalence analysis. Groups not in the active set
    /// are ignored, producing coarser (but faster) token classes.
    pub fn build_with_group_filter(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        vocab: &crate::Vocab,
        disallowed_follows: &std::collections::BTreeMap<u32, crate::ds::bitset::BitSet>,
        ignore_terminal: Option<u32>,
        active_groups: Option<&[bool]>,
    ) -> Self {
        combined::analyze_equivalences_with_group_filter(
            tokenizer, vocab, disallowed_follows, ignore_terminal, active_groups,
        )
    }

    /// L1 fast path: compute equivalence classes using direct DFA fingerprinting.
    /// Only valid when all terminals have max path length ≤ 1.
    pub fn build_l1(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        vocab: &crate::Vocab,
    ) -> Self {
        combined::analyze_equivalences_l1_fast(tokenizer, vocab)
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

/// Merge two id_maps that cover the **same** states and tokens (e.g., L1 + L2+
/// within a single partition).  The result is the finest common refinement:
/// two items share a class only if they share a class in **both** inputs.
///
/// Returns `(merged_id_map, l1_to_merged_tsid_map, l2p_to_merged_tsid_map,
///            l1_to_merged_token_map, l2p_to_merged_token_map)`.
pub fn merge_overlapping_id_maps(
    l1_map: &InternalIdMap,
    l2p_map: &InternalIdMap,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> (
    InternalIdMap,
    Vec<u32>,  // l1 internal tsid → merged internal tsid
    Vec<u32>,  // l2p internal tsid → merged internal tsid
    Vec<u32>,  // l1 internal token → merged internal token
    Vec<u32>,  // l2p internal token → merged internal token
) {
    use std::collections::HashMap;

    // --- State refinement: composite key = (l1_class, l2p_class) ---
    let mut composite_to_class: HashMap<(u32, u32), u32> = HashMap::new();
    let mut state_original_to_internal = vec![0u32; num_tokenizer_states];
    let mut state_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut state_representative_ids: Vec<u32> = Vec::new();

    for state in 0..num_tokenizer_states {
        let l1_class = l1_map.tokenizer_states.original_to_internal[state];
        let l2p_class = l2p_map.tokenizer_states.original_to_internal[state];
        let key = (l1_class, l2p_class);
        let next_id = state_internal_to_originals.len() as u32;
        let class = *composite_to_class.entry(key).or_insert_with(|| {
            state_internal_to_originals.push(Vec::new());
            state_representative_ids.push(state as u32);
            next_id
        });
        state_original_to_internal[state] = class;
        state_internal_to_originals[class as usize].push(state as u32);
    }

    // Build l1/l2p internal tsid → merged tsid maps.
    let mut l1_tsid_to_merged = vec![u32::MAX; l1_map.tokenizer_states.internal_to_originals.len()];
    let mut l2p_tsid_to_merged = vec![u32::MAX; l2p_map.tokenizer_states.internal_to_originals.len()];
    for (state, &merged_class) in state_original_to_internal.iter().enumerate() {
        let l1_class = l1_map.tokenizer_states.original_to_internal[state];
        let l2p_class = l2p_map.tokenizer_states.original_to_internal[state];
        l1_tsid_to_merged[l1_class as usize] = merged_class;
        l2p_tsid_to_merged[l2p_class as usize] = merged_class;
    }

    // --- Token refinement: composite key = (l1_class, l2p_class) ---
    let mut token_composite_to_class: HashMap<(u32, u32), u32> = HashMap::new();
    let mut token_original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut token_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut token_representative_ids: Vec<u32> = Vec::new();

    // Build l1/l2p internal token → merged token maps.
    let l1_num_tokens = l1_map.vocab_tokens.internal_to_originals.len();
    let l2p_num_tokens = l2p_map.vocab_tokens.internal_to_originals.len();
    let mut l1_token_to_merged = vec![u32::MAX; l1_num_tokens];
    let mut l2p_token_to_merged = vec![u32::MAX; l2p_num_tokens];

    for token_id in 0..=max_token_id {
        let l1_class = l1_map.vocab_tokens.original_to_internal.get(token_id as usize).copied().unwrap_or(u32::MAX);
        let l2p_class = l2p_map.vocab_tokens.original_to_internal.get(token_id as usize).copied().unwrap_or(u32::MAX);
        if l1_class == u32::MAX && l2p_class == u32::MAX {
            continue;
        }
        let key = (l1_class, l2p_class);
        let next_id = token_internal_to_originals.len() as u32;
        let class = *token_composite_to_class.entry(key).or_insert_with(|| {
            token_internal_to_originals.push(Vec::new());
            token_representative_ids.push(token_id);
            next_id
        });
        token_original_to_internal[token_id as usize] = class;
        token_internal_to_originals[class as usize].push(token_id);
        if l1_class != u32::MAX {
            l1_token_to_merged[l1_class as usize] = class;
        }
        if l2p_class != u32::MAX {
            l2p_token_to_merged[l2p_class as usize] = class;
        }
    }

    let merged = InternalIdMap {
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
    };

    (merged, l1_tsid_to_merged, l2p_tsid_to_merged, l1_token_to_merged, l2p_token_to_merged)
}
