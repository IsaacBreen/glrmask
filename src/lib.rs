#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This crate root is the small glrmask counterpart to sep1's broader `lib.rs`, re-exporting the main compile/runtime entrypoints after the crate-root thinning pass.

#![deny(warnings)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
pub(crate) mod import;
pub(crate) mod runtime;
mod vocab;

pub use compiler::debug::{AutomataDebug, CompileDebug, TerminalDebug};
pub use error::{GlrMaskError, Result};
pub use runtime::{Constraint, ConstraintState};
pub use vocab::Vocab;

