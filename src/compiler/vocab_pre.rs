//! Vocabulary preprocessing.
//!
//! Computes token-set IDs (TSIDs), equivalence classes, and token-to-TSID mappings.

use crate::Vocab;

/// Token-set equivalence class mapping.
///
/// Maps each token ID to a token-set ID. Tokens that behave identically
/// through the DWA get the same TSID.
#[derive(Debug, Clone)]
pub struct VocabMapping {
    /// `token_to_tsid[token_id]` = TSID.
    pub token_to_tsid: Vec<u32>,
    /// Number of unique TSIDs.
    pub num_tsids: u32,
}

impl VocabMapping {
    /// Compute vocabulary equivalence classes.
    pub fn compute(_vocab: &Vocab) -> Self {
        // TODO: Implement
        Self {
            token_to_tsid: Vec::new(),
            num_tsids: 0,
        }
    }
}
