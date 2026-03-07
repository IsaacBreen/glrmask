//! Compiler-side equivalence analysis.
//!
//! This stage family compacts original tokenizer-state IDs and original
//! vocab-token IDs into narrower internal ID spaces. Multiple originals may
//! share one internal ID when they are equivalent for compiler purposes.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub mod combined;
pub mod state_analysis;
pub mod vocab_analysis;

/// A many-original-to-one-internal ID mapping.
#[derive(Debug, Clone)]
pub struct ManyToOneIdMap {
    /// `original_to_internal[original]` = compact internal ID, or `u32::MAX`
    /// when the original ID is not represented.
    pub original_to_internal: Vec<u32>,
    /// `internal_to_originals[internal]` = all original IDs that collapse to
    /// that internal ID.
    pub internal_to_originals: Vec<Vec<u32>>,
}

impl ManyToOneIdMap {
    /// Number of compact internal IDs in this mapping.
    pub fn num_internal_ids(&self) -> u32 {
        self.internal_to_originals.len() as u32
    }

    /// Largest original ID represented by this mapping, or 0 when empty.
    pub fn max_original_id(&self) -> u32 {
        self.original_to_internal
            .len()
            .checked_sub(1)
            .map(|i| i as u32)
            .unwrap_or(0)
    }
}

/// Compiler-side joint internal ID mappings.
#[derive(Debug, Clone)]
pub struct InternalIdMap {
    /// Compact mapping for tokenizer DFA state IDs.
    pub tokenizer_states: ManyToOneIdMap,
    /// Compact mapping for original vocab / LLM token IDs.
    pub vocab_tokens: ManyToOneIdMap,
}

impl InternalIdMap {
    /// Build the joint mapping directly from the tokenizer and vocab.
    pub fn build(tokenizer: &crate::automata::lexer::tokenizer::Tokenizer, vocab: &crate::Vocab) -> Self {
        combined::analyze_equivalences(tokenizer, vocab)
    }

    /// Number of compact tokenizer-state IDs.
    pub fn num_tsids(&self) -> u32 {
        self.tokenizer_states.num_internal_ids()
    }

    /// Largest original vocab token ID represented by the mapping.
    pub fn max_token_id(&self) -> u32 {
        self.vocab_tokens.max_original_id()
    }
}

pub(crate) use combined::analyze_equivalences;
