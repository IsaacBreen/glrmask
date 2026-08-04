#![deny(warnings)]
#![allow(dead_code)]

use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Compilation error: {0}")]
    Compilation(String),
}

pub(crate) type GlrMaskError = Error;

pub(crate) mod ds {
    pub(crate) use glrmask_weight::__private as weight;
}

mod automata {
    pub mod weighted_u32;
}

pub use automata::weighted_u32;
pub use weighted_u32 as weighted;
