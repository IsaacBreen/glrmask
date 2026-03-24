use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type GlrMaskError = Error;

pub type Result<T> = std::result::Result<T, Error>;
