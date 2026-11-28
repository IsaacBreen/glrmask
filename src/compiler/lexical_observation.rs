//! Source-pinned lexical observations for a single compile.
//! The raw byte action, exact vocabulary, conservative global partition,
//! classifier, and optional certified local synthesis share this owner.
//! Local and terminal-conditioned partitions may refine or forget different
//! observations; no finite-token proof authorizes unrestricted byte steps.
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use crate::automata::lexer::Lexer;
use crate::automata::lexer::observation::ByteRows;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::constraint_possible_matches as cpm;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::{
    build_global_max_length_state_map,
    build_terminal_dwa_families_with_precomputed_global_max_length,
    build_vocab_equivalence_partition_with_precomputed_global_max_length,
    build_vocab_partition_from_static_id_maps,
    classify::{SharedClassifyBytesets, SharedClassifyCache},
    types::{TerminalColoring, TerminalDwaFamilies, TerminalDwaPhaseProfile},
    PartitionLocalSynthesisPlan, PreparedPartitionLocalTokenizers,
};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

pub(crate) struct LexicalObservationSource<'v> {
    pub tokenizer: Tokenizer,
    pub flat_trans: ByteRows,
    vocab: &'v Vocab,
    global: OnceLock<ManyToOneIdMap>,
    classification: SharedClassifyCache,
    local_synthesis: Option<Arc<PartitionLocalSynthesisPlan>>,
    prepared_locals: Option<Arc<PreparedPartitionLocalTokenizers>>,
}

impl<'v> LexicalObservationSource<'v> {
    pub fn new(
        tokenizer: Tokenizer,
        vocab: &'v Vocab,
        local_synthesis: Option<Arc<PartitionLocalSynthesisPlan>>,
        prepared_locals: Option<Arc<PreparedPartitionLocalTokenizers>>,
    ) -> Self {
        let flat_trans = ByteRows::from_tokenizer(&tokenizer);
        Self {
            tokenizer, flat_trans, vocab, local_synthesis, prepared_locals,
            global: OnceLock::new(),
            classification: SharedClassifyCache::new(),
        }
    }

    pub fn global_map(&self) -> &ManyToOneIdMap {
        self.global.get_or_init(|| build_global_max_length_state_map(&self.tokenizer, self.vocab))
    }

    fn physical_classes(&self) -> &[u8; 256] {
        &self.flat_trans.byte_to_column
    }

    pub fn classification(&self, num_terminals: u32) -> &SharedClassifyCache {
        self.classification.get_or_init(|| SharedClassifyBytesets::build_with_physical_classes(&self.tokenizer, num_terminals, *self.physical_classes()));
        &self.classification
    }

    pub fn terminal_families(
        &self,
        coloring: &TerminalColoring,
        use_coloring: bool,
        ignore: Option<TerminalID>,
        grammar: &AnalyzedGrammar,
        disallowed: &BTreeMap<u32, BitSet>,
        always_allowed: Option<&[Vec<TerminalID>]>,
    ) -> (TerminalDwaFamilies, TerminalDwaPhaseProfile) {
        build_terminal_dwa_families_with_precomputed_global_max_length(
            &self.tokenizer, self.vocab, coloring, use_coloring, ignore, grammar,
            disallowed, always_allowed, self.flat_trans.clone(),
            self.global_map(), Some(self.classification(grammar.num_terminals)),
            self.local_synthesis.as_deref(), self.prepared_locals.as_deref(), None,
        )
    }

    pub fn vocab_partition(
        &self,
        dedicated: bool,
        ignore: Option<TerminalID>,
        grammar: &AnalyzedGrammar,
        disallowed: &BTreeMap<u32, BitSet>,
    ) -> ManyToOneIdMap {
        if dedicated {
            build_vocab_equivalence_partition_with_precomputed_global_max_length(
                &self.tokenizer, self.vocab, ignore, grammar, disallowed,
                self.flat_trans.clone(), self.global_map(),
                self.local_synthesis.as_deref(),
                Some(self.classification(self.tokenizer.num_terminals())),
            )
        } else {
            build_vocab_partition_from_static_id_maps(
                &self.tokenizer, self.vocab,
                &TerminalColoring::identity(grammar.num_terminals as usize), false,
                ignore, grammar, disallowed, None, self.flat_trans.clone(),
                self.global_map(), Some(&self.classification),
                self.local_synthesis.as_deref(), self.prepared_locals.as_deref(),
            )
        }
    }

    pub fn possible_matches(&self, eager: bool) -> cpm::ConstraintPossibleMatchesComputation {
        if eager {
            cpm::compute_constraint_possible_matches_for_vocab_with_byte_observations(
                &self.tokenizer, self.vocab, cpm::ConstraintPossibleMatchesConfig::EAGER,
                &self.flat_trans,
            )
        } else {
            cpm::compute_constraint_possible_matches_for_vocab(
                &self.tokenizer, self.vocab, cpm::ConstraintPossibleMatchesConfig::DEFER_TO_DYNAMIC_MASK,
            )
        }
    }
}
