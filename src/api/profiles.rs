//! Public profiling and trace types.
//!
//! The core API is mathematical: mask and commit.  These profile structs are
//! exposed because publication benchmarks and downstream integrations need to
//! attribute runtime cost to the corresponding algorithmic phases.  They are not
//! required for normal generation.

#[doc(inline)]
pub use crate::runtime::{CommitProfile, GssProfileSummary, MaskProfile, PerAdvanceEntry};

/// Parser-advance trace and profile types.
///
/// These are lower-level than [`MaskProfile`] and [`CommitProfile`].  They remain
/// public because the Python bindings and analysis harnesses inspect individual
/// GLR advance waves.
pub mod advance {
    #[doc(inline)]
    pub use crate::runtime::{
        AdvanceProfile,
        AdvanceTrace,
        AdvanceTraceGoto,
        AdvanceTraceReduce,
        AdvanceTraceStep,
        AdvanceTraceWave,
    };
}

#[doc(inline)]
pub use advance::{
    AdvanceProfile,
    AdvanceTrace,
    AdvanceTraceGoto,
    AdvanceTraceReduce,
    AdvanceTraceStep,
    AdvanceTraceWave,
};
