#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub mod combined;
pub mod compat;
pub mod state;
pub mod vocab;
pub mod combined_equivalence_analysis;

#[derive(Debug, Clone)]
pub struct ManyToOneIdMap {
    pub original_to_internal: Vec<u32>,
    pub internal_to_originals: Vec<Vec<u32>>,
}

impl ManyToOneIdMap {
    pub fn num_internal_ids(&self) -> u32 {
        self.internal_to_originals.len() as u32
    }

    pub fn internal_id_for_original(&self, original_id: u32) -> Option<u32> {
        self.original_to_internal
            .get(original_id as usize)
            .copied()
            .filter(|internal_id| *internal_id != u32::MAX)
    }

    pub fn representative_original_id_for_internal(&self, internal_id: u32) -> Option<u32> {
        self.internal_to_originals
            .get(internal_id as usize)
            .and_then(|original_ids| original_ids.first())
            .copied()
    }

    pub fn representative_original_id_for_original(&self, original_id: u32) -> Option<u32> {
        self.internal_id_for_original(original_id)
            .and_then(|internal_id| self.representative_original_id_for_internal(internal_id))
    }

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
    pub fn build(tokenizer: &crate::automata::lexer::tokenizer::Tokenizer, vocab: &crate::Vocab) -> Self {
        combined::analyze_equivalences(tokenizer, vocab)
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

    pub fn max_token_id(&self) -> u32 {
        self.vocab_tokens.max_original_id()
    }
}

pub(crate) use combined::analyze_equivalences;
