//! Projected-residual L1 construction and its shared input.

mod common;
mod projected;
mod support;

use crate::automata::lexer::observation::ByteRows;
pub(crate) use projected::{build_projected_vocab_equivalence, prepare_finite_vocab_projection};


use super::L1IdentityVocabOrder;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::grammar::flat::TerminalID;
use crate::terminal_dwa::l2p::equivalence_analysis::state_equivalence::nfa::{
    TokenBoundedAnalysisTopology, TokenBoundedAnalysisTrie,
};
use crate::terminal_dwa::types::{LocalIdMapTerminalDwa, TerminalColoring};
use crate::{Vocab, compiler::glr::analysis::AnalyzedGrammar};

#[derive(Clone, Copy)]
pub struct BuildInput<'a> {
    pub partition_label: &'a str,
    pub tokenizer: &'a Tokenizer,
    pub vocab: &'a Vocab,
    pub terminal_coloring: &'a TerminalColoring,
    pub use_terminal_coloring: bool,
    pub ignore_terminal: Option<TerminalID>,
    pub grammar: &'a AnalyzedGrammar,
    pub active_terminals: &'a [bool],
    pub flat_trans: &'a ByteRows,
    pub transitions_by_byte: Option<&'a crate::terminal_dwa::classify::ByteTransitionIndex>,
    pub initial_state_map: Option<&'a ManyToOneIdMap>,
    /// `u32::MAX` in `initial_state_map` means excluded from the complete
    /// token-start domain, rather than an unclassified raw state that an L1
    /// implementation should conservatively reintroduce.
    pub initial_state_domain_is_exact: bool,
    pub shared_generic_nfa_topology: Option<&'a TokenBoundedAnalysisTopology>,
    pub shared_generic_nfa_trie: Option<&'a TokenBoundedAnalysisTrie>,
    pub subset_parent_order: Option<&'a L1IdentityVocabOrder>,
    pub id_map_only: bool,
}

pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    projected::build(input)
}
