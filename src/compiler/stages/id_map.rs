//! Internal ID mapping for compiler artifacts.
//!
//! Compacts original tokenizer-state IDs and original vocab-token IDs into
//! narrower internal ID spaces. The mapping is many-original-to-one-internal:
//! multiple original IDs may share one internal ID when they are equivalent
//! for compiler purposes.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::TokenizerDfa;

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

/// Compiler-side internal ID mappings.
#[derive(Debug, Clone)]
pub struct InternalIdMap {
    /// Compact mapping for tokenizer DFA state IDs.
    pub tokenizer_states: ManyToOneIdMap,
    /// Compact mapping for original vocab / LLM token IDs.
    pub vocab_tokens: ManyToOneIdMap,
}

impl InternalIdMap {
    /// Build compiler-side internal ID mappings from the tokenizer and vocab.
    pub fn build(tokenizer: &TokenizerDfa, vocab: &Vocab) -> Self {
        let tokenizer_states = ManyToOneIdMap {
            original_to_internal: (0..tokenizer.dfa.num_states() as u32).collect(),
            internal_to_originals: (0..tokenizer.dfa.num_states() as u32)
                .map(|state| vec![state])
                .collect(),
        };

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

        let vocab_tokens = ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
        };

        Self {
            tokenizer_states,
            vocab_tokens,
        }
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

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::TokenizerDfa;
    use crate::compiler::grammar::ast::{GrammarDef, Rule, Symbol, TerminalDef};

    #[test]
    fn test_internal_id_map_shape() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![TerminalDef {
                id: 0,
                name: "a".into(),
                pattern: "a".into(),
            }],
        };
        let tok = TokenizerDfa::from_grammar_def(&gdef);
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
            ],
            None,
        );
        let id_map = InternalIdMap::build(&tok, &vocab);

        assert!(id_map.num_tsids() >= 1);
        assert_eq!(id_map.max_token_id(), 2);
    }
}
