use thiserror::Error as ThisError;

#[derive(Debug)]
pub(crate) struct InternalInvariantViolation {
    message: String,
}

pub(crate) fn fail_internal_invariant(message: impl Into<String>) -> ! {
    // `resume_unwind` deliberately skips the global panic hook. The payload is
    // caught at the public compilation boundary and converted into a normal,
    // structured `Error::InternalInvariant`; unrelated panics still propagate.
    std::panic::resume_unwind(Box::new(InternalInvariantViolation {
        message: message.into(),
    }))
}

pub(crate) fn catch_internal_invariant<T>(f: impl FnOnce() -> T) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<InternalInvariantViolation>() {
            Ok(violation) => Err(Error::InternalInvariant(violation.message)),
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_invariant_crossing_rayon_is_returned_as_a_normal_error() {
        let error = catch_internal_invariant(|| {
            let _: ((), usize) = rayon::join(
                || fail_internal_invariant("analysis coordinate escaped its domain"),
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
