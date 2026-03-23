#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

#![deny(warnings)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
pub(crate) mod grammar;
pub(crate) mod import;
pub(crate) mod runtime;
mod vocab;

pub use compiler::debug::{AutomataDebug, CompileDebug, TerminalDebug};
pub use ds::weight::{
    clear_all_weights,
    clear_stale_weights,
    clear_weight_caches,
    clear_weight_op_caches,
};
pub use error::{GlrMaskError, Result};
pub use runtime::{CommitDebugMetrics, CommitDebugTrace, Constraint, ConstraintState, ConstraintStateSummary, MaskDebugMetrics};
pub use vocab::Vocab;

#[doc(hidden)]
pub fn __check_live_minimal_tokenizer_fineness() {
    compiler::stages::equivalence_analysis::combined_equivalence_analysis::check_live_minimal_tokenizer_fineness();
}
