mod artifact;
mod commit;
mod constraint;
mod dynamic_mask;
mod finalize;
mod mask;
pub(crate) mod mask_mapping;
mod serde;
mod state;
mod token_space;
pub(crate) use glrmask_artifact::CommitTemplateDfas;
pub(crate) use artifact::{
    ConstraintRuntimeBackend, DynamicMaskTrie, DynamicMaskVocab, FastCommitTemplateDfas,
    FastTokenizerTransitions, SegmentedBoundaryParser, SegmentedParserComponent,
    SpecialTokenTerminal, StaticDynamicOverlayMetadata,
};
#[allow(unused_imports)]
pub use crate::compiler::glr::parser::{AdvanceTrace, AdvanceTraceStep};
#[allow(unused_imports)]
pub use commit::profile::{CommitProfile, GssProfileSummary, PerAdvanceEntry};
pub use constraint::Constraint;
pub(crate) use constraint::TokenMaskCachePrebuild;
#[allow(unused_imports)]
pub use mask::profile::MaskProfile;
pub use state::ConstraintState;

pub(crate) fn initialize_hot_path_config() {
    mask::initialize_runtime_config();
    commit::initialize_runtime_config();
}
