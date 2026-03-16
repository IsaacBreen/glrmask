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
pub(crate) mod runtime;
mod vocab;

pub use error::{GlrMaskError, Result};
pub use runtime::{
    Constraint,
    ConstraintState,
    ConstraintStateSnapshot,
    ConstraintStateSnapshotEntry,
    ConstraintStateSummary,
    MaskDebugMetrics,
};
pub use vocab::Vocab;
