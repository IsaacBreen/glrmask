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
pub use error::{Error, GlrMaskError, Result};
pub use runtime::{
    Constraint,
    ConstraintState,
};
pub use vocab::Vocab;
