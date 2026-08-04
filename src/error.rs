use thiserror::Error as ThisError;

pub(crate) fn catch_internal_invariant<T>(f: impl FnOnce() -> T) -> Result<T> {
    glrmask_invariant::__private::catch_internal_invariant_message(f).map_err(Error::InternalInvariant)
}

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Internal compiler invariant violated: {0}")]
    InternalInvariant(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type GlrMaskError = Error;

pub type Result<T> = std::result::Result<T, Error>;

impl From<glrmask_grammar::Error> for Error {
    fn from(error: glrmask_grammar::Error) -> Self {
        match error {
            glrmask_grammar::Error::GrammarParse(message) => Self::GrammarParse(message),
        }
    }
}

impl From<glrmask_weighted_automata::Error> for Error {
    fn from(error: glrmask_weighted_automata::Error) -> Self {
        match error {
            glrmask_weighted_automata::Error::Compilation(message) => Self::Compilation(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_invariant_crossing_rayon_is_returned_as_a_normal_error() {
        let error = catch_internal_invariant(|| {
            let _: ((), usize) = rayon::join(
                || glrmask_invariant::__private::fail_internal_invariant("analysis coordinate escaped its domain"),
                || 1,
            );
        })
        .expect_err("the internal invariant payload must cross the Rayon boundary");

        assert!(matches!(error, Error::InternalInvariant(_)));
        assert_eq!(
            error.to_string(),
            "Internal compiler invariant violated: analysis coordinate escaped its domain"
        );
    }

    #[test]
    fn unrelated_panics_are_not_reclassified_as_compiler_errors() {
        let panic = std::panic::catch_unwind(|| {
            let _ = catch_internal_invariant(|| {
                std::panic::resume_unwind(Box::new("unrelated panic payload"))
            });
        });

        assert!(panic.is_err());
    }
}
