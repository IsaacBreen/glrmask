//! Runtime artifact, state, Mask, and Commit.
//!
//! The runtime side of the crate is deliberately split along the paper's two
//! online operations:
//!
//! - **Mask** reads a [`ConstraintState`] and writes a vocabulary bitset.
//! - **Commit** consumes accepted bytes/tokens and mutates the
//!   [`ConstraintState`] frontier.
//!
//! Both operations borrow the immutable [`Constraint`] artifact.  Neither owns
//! compile-time construction logic.

mod artifact;
mod bitmask_ops;
mod commit;
mod constraint;
mod mask;
mod token_space;
mod state;
mod template_dfa;

pub(crate) use artifact::{CommitTemplateDfas, CompiledArtifactParts, TemplateDfasByTerminal};
pub use crate::parser::glr::advance::{
    AdvanceProfile,
    AdvanceTrace,
    AdvanceTraceGoto,
    AdvanceTraceReduce,
    AdvanceTraceStep,
    AdvanceTraceWave,
};
pub use commit::profile::{CommitProfile, GssProfileSummary, PerAdvanceEntry};
pub use constraint::Constraint;
pub use mask::profile::MaskProfile;
pub use token_space::final_mask_mapping::FinalMaskMapping;
pub use state::ConstraintState;
