#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: There is no meaningful sep1 crate-level equivalent for this file; sep1 mostly uses ad hoc `Result<_, String>`-style errors instead of one central public error enum.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GlrMaskError {
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, GlrMaskError>;
