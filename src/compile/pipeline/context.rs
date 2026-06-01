//! Typed intermediate objects for the compile phase graph.
//!
//! The point of this file is not to add abstraction for its own sake.  It makes
//! the compile graph explicit: each phase consumes a named input object and
//! produces a named output object.  That prevents the old `pipeline.rs` failure
//! mode where a local variable's lifetime was the only indication of the
//! underlying mathematical dependency.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compile::scan_relation::{RuntimeCanMatchByTerminal, ScanRelationComputation};
use crate::compile::terminal_dwa::classify::SharedClassifyCache;
use crate::compile::terminal_dwa::types::TerminalColoring;
use crate::parser::glr::analysis::AnalyzedGrammar;
use crate::parser::glr::table::GLRTable;
use crate::compile::id_space::{InternalIdMap, ManyToOneIdMap};
use crate::compile::mapped_artifact::MappedArtifact;
use crate::compile::template_dfa::Templates;
use crate::sets::bitset::BitSet;
use crate::grammar::flat::GrammarDef;
use crate::runtime::TemplateDfasByTerminal;

/// Input to the phase graph after frontend parsing but before normalization.
pub(crate) struct OwnedCompileInput<'vocab> {
    pub(crate) grammar: GrammarDef,
    pub(crate) vocab: &'vocab Vocab,
}

/// Input to the expensive compile graph after grammar normalization.
pub(crate) struct PreparedCompileInput<'vocab> {
    pub(crate) prepared_grammar: GrammarDef,
    pub(crate) vocab: &'vocab Vocab,
}

/// Outputs of tokenizer construction, grammar analysis, GLR table construction,
/// and derived parser/terminal facts.
pub(crate) struct GrammarAnalysisOutput {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) analyzed_grammar: AnalyzedGrammar,
    pub(crate) table: GLRTable,
    pub(crate) terminal_coloring: TerminalColoring,
    pub(crate) disallowed_follows: BTreeMap<u32, BitSet>,
}

/// Shared precomputation used by both Terminal-DWA construction and scan relation work.
pub(crate) struct TerminalScanSupport {
    pub(crate) shared_classify_cache: SharedClassifyCache,
    pub(crate) flat_transitions: Arc<[u32]>,
    pub(crate) global_max_length_state_map: ManyToOneIdMap,
}

/// Coupled outputs of the two token-byte interpretations.
///
/// `terminal_dwa` is about completed terminal sequences.  `scan_relation` is
/// about boundary states and possible completions for partial token scans.
/// They are built from the same tokenizer/vocab facts but they intentionally do
/// not share an equivalence proof until the reconciliation phase.
pub(crate) struct TerminalAndScanOutput {
    pub(crate) terminal_dwa: MappedArtifact<DWA>,
    pub(crate) scan_relation: ScanRelationComputation,
}

/// Template recognizers used by Parser-DWA construction and commit-time fast paths.
pub(crate) struct TemplateOutput {
    pub(crate) templates: Templates,
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
}

/// Outputs of Parser-DWA construction and coordinate reconciliation.
pub(crate) struct ReconciledArtifacts {
    pub(crate) parser_dwa: DWA,
    pub(crate) can_match: RuntimeCanMatchByTerminal,
    pub(crate) internal_ids: InternalIdMap,
    pub(crate) parser_dwa_interned_ranges: usize,
    pub(crate) can_match_interned_ranges: usize,
    pub(crate) parser_can_match_joint_interned_ranges: usize,
    pub(crate) terminal_dwa_interned_ranges_before_can_match_reconcile: usize,
    pub(crate) can_match_interned_ranges_before_can_match_reconcile: usize,
    pub(crate) terminal_can_match_joint_interned_ranges_before_reconcile: usize,
    pub(crate) terminal_can_match_joint_interned_ranges: usize,
}
