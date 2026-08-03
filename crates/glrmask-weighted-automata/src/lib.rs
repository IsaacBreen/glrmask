#![deny(warnings)]
#![allow(dead_code)]

use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Compilation error: {0}")]
    Compilation(String),
}

pub type GlrMaskError = Error;

pub mod ds {
    pub use glrmask_weight as weight;
}

pub mod automata {
    pub mod weighted_u32;
    pub use weighted_u32 as weighted;
}

pub use automata::weighted_u32;
