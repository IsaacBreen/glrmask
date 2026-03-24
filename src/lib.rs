#![deny(warnings)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
pub(crate) mod grammar;
pub(crate) mod import;
pub(crate) mod runtime;
mod vocab;

pub use ds::weight::{
    clear_all_weights,
    clear_stale_weights,
    clear_weight_caches,
    clear_weight_op_caches,
};
pub use error::{Error, Result};
#[allow(deprecated)]
pub use error::GlrMaskError;
pub use runtime::{
    CommitMetrics,
    CommitTrace,
    Constraint,
    ConstraintState,
    ConstraintStateSummary,
    MaskMetrics,
};
pub use vocab::Vocab;

#[doc(hidden)]
pub fn __check_live_minimal_tokenizer_fineness() {
    compiler::stages::equivalence_analysis::combined_equivalence_analysis::check_live_minimal_tokenizer_fineness();
}
