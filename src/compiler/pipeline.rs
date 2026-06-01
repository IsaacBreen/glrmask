//! Deprecated compatibility shim for the old pipeline path.
//!
//! The compile pipeline now lives at [`crate::compile::pipeline`].  This file is
//! intentionally tiny so the remaining `compiler` namespace is about GLR/parser
//! internals rather than publication-facing compile objects.

#[allow(unused_imports)]
pub(crate) use crate::compile::pipeline::{
    COMPILE_PHASE_ORDER,
    CompilePhase,
    compile_owned,
    compile_owned_profiled,
};
#[allow(unused_imports)]
pub(crate) use crate::compile::profiling::{
    CompilePhaseProfile,
    compile_profile_enabled,
    compile_profile_summary_enabled,
    emit_compile_profile_summary,
};
