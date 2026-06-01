//! Backward-compatible compile facade.
//!
//! New code should use `crate::compile::pipeline` and `crate::compile::profiling`.
//! This module remains during the publication cleanup so existing tests and
//! frontends do not need to move in the same patch.

#[allow(unused_imports)]
pub(crate) use crate::compile::pipeline::{compile_owned, compile_owned_profiled};
#[allow(unused_imports)]
pub(crate) use crate::compile::profiling::{
    compile_profile_enabled,
    emit_compile_profile_summary,
};

pub(crate) fn prepare_vocab_for_compile(vocab: &crate::Vocab) {
    crate::compile::terminal_dwa::prepare_vocab_for_terminal_dwa(vocab);
    crate::compile::scan_relation::prepare_vocab_for_scan_relation(vocab);
}
