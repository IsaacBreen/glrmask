mod artifact;
mod commit;
mod constraint;
mod dynamic_mask;
mod finalize;
mod mask;
pub(crate) mod mask_mapping;
mod serde;
pub(crate) use serde::compact_large_non_dwa_weight_runtime;
mod state;
mod token_space;
pub(crate) use glrmask_artifact::CommitTemplateDfas;
#[allow(unused_imports)]
pub(crate) use artifact::{
    dynamic_mask_vocab_layout_class, BoundaryTerminalNwa, BoundaryTerminalNwaNode,
    BoundaryTerminalNwaTransition, BoundaryTerminalTrieNode, CompositionGrammarSummary,
    ConstraintRuntimeBackend, DynamicMaskTrie, DynamicMaskVocab, FastCommitTemplateDfas,
    FastTokenizerTransitions, SegmentedBoundaryParser, SegmentedBoundaryTerminalTrie,
    SegmentedParserComponent, LateGrammarSlot,
    SpecialTokenTerminal, StaticDynamicOverlayMetadata,
};
pub(crate) use artifact::token_bytes_artifact_serde::PackedTokenBytes;
#[allow(unused_imports)]
pub use crate::compiler::glr::parser::{AdvanceTrace, AdvanceTraceStep};
#[allow(unused_imports)]
pub use commit::profile::{CommitProfile, GssProfileSummary, PerAdvanceEntry};
pub use constraint::Constraint;
pub(crate) use constraint::{InternalTokenMaskPrebuild, RuntimeWeightRef, TokenMaskCachePrebuild};
#[allow(unused_imports)]
pub use mask::profile::MaskProfile;
pub use state::ConstraintState;

pub(crate) fn initialize_hot_path_config() {
    mask::initialize_runtime_config();
    commit::initialize_runtime_config();
}
