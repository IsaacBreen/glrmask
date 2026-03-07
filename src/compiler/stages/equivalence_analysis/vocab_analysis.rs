//! Vocab-token-side equivalence analysis.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::Vocab;
use crate::compiler::stages::equivalence_analysis::ManyToOneIdMap;

/// Analyze vocab-token equivalence classes for compiler use.
pub(crate) fn analyze_vocab_equivalences(vocab: &Vocab) -> ManyToOneIdMap {
    let max_token_id = vocab
        .entries
        .iter()
        .map(|(token_id, _)| *token_id)
        .max()
        .unwrap_or(0);
    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut interned: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();

    for (token_id, bytes) in &vocab.entries {
        let internal_id = if let Some(&existing) = interned.get(bytes) {
            existing
        } else {
            let next = internal_to_originals.len() as u32;
            interned.insert(bytes.clone(), next);
            internal_to_originals.push(Vec::new());
            next
        };

        if let Some(slot) = original_to_internal.get_mut(*token_id as usize) {
            *slot = internal_id;
        }
        internal_to_originals[internal_id as usize].push(*token_id);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
    }
}
