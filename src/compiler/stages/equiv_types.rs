//! Shared equivalence-analysis types used across multiple compiler stages.
//!
//! `ManyToOneIdMap` and `InternalIdMap` are pure data types that represent
//! equivalence-class mappings.  The analysis that *produces* them lives in
//! `id_map_and_terminal_dwa::l2p::equivalence_analysis`.

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

    pub fn from_original_to_internal_allowing_unmapped(
        original_to_internal: Vec<u32>,
        num_internal: u32,
    ) -> Self {
        let mut internal_to_originals = vec![Vec::new(); num_internal as usize];
        let mut representative_original_ids = vec![u32::MAX; num_internal as usize];
        for (original, &internal) in original_to_internal.iter().enumerate() {
            if internal == u32::MAX || (internal as usize) >= internal_to_originals.len() {
                continue;
            }
            let originals = &mut internal_to_originals[internal as usize];
            if originals.is_empty() {
                representative_original_ids[internal as usize] = original as u32;
            }
            originals.push(original as u32);
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
