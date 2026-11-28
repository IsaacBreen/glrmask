//! A single physical lexer coordinate and its derived compiler indexes.
//! The owner binds the flat byte action and derived observations to the same
//! immutable tokenizer. Finite-vocabulary quotients are separate proofs;
//! their rows must not be used as unrestricted runtime transitions.
use std::sync::{Arc, OnceLock};

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::constraint_possible_matches as cpm;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::{
    build_global_max_length_state_map, l1,
    classify::{SharedClassifyBytesets, SharedClassifyCache},
    l2p::equivalence_analysis::compat::compute_physical_byte_classes,
};
use crate::Vocab;

pub(super) struct LexicalObservationSource {
    pub tokenizer: Tokenizer,
    pub flat_trans: Arc<[u32]>,
    physical_classes: OnceLock<[u8; 256]>,
}

impl LexicalObservationSource {
    pub fn new(tokenizer: Tokenizer) -> Self {
        let flat_trans = Arc::from(l1::build_flat_transition_table(&tokenizer));
        Self { tokenizer, flat_trans, physical_classes: OnceLock::new() }
    }

    pub fn global_map(&self, vocab: &Vocab) -> ManyToOneIdMap {
        build_global_max_length_state_map(&self.tokenizer, vocab)
    }

    fn physical_classes(&self) -> &[u8; 256] {
        self.physical_classes.get_or_init(|| compute_physical_byte_classes(&self.tokenizer, &self.flat_trans))
    }

    pub fn classification(&self, num_terminals: u32) -> SharedClassifyCache {
        let cache = SharedClassifyCache::new();
        cache.get_or_init(|| SharedClassifyBytesets::build_with_physical_classes(&self.tokenizer, num_terminals, *self.physical_classes()));
        cache
    }

    pub fn possible_matches(&self, vocab: &Vocab, eager: bool) -> cpm::ConstraintPossibleMatchesComputation {
        if eager {
            let classes = self.physical_classes();
            cpm::compute_constraint_possible_matches_for_vocab_with_byte_observations(
                &self.tokenizer, vocab, cpm::ConstraintPossibleMatchesConfig::EAGER,
                classes, &self.flat_trans,
            )
        } else {
            cpm::compute_constraint_possible_matches_for_vocab(
                &self.tokenizer, vocab, cpm::ConstraintPossibleMatchesConfig::DEFER_TO_DYNAMIC_MASK,
            )
        }
    }
}
