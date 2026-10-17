use thiserror::Error as ThisError;

pub(crate) fn catch_internal_invariant<T>(f: impl FnOnce() -> T) -> Result<T> {
    glrmask_invariant::__private::catch_compilation_resource_limit(|| {
        glrmask_invariant::__private::catch_internal_invariant_message(f)
    })
    .map_err(|message| Error::Compilation(format!("resource limit exceeded: {message}")))?
    .map_err(Error::InternalInvariant)
}

/// Parse, build, artifact or sequence-state failure. Source coordinates are included when the frontend can locate the error.
#[derive(ThisError, Debug, Clone)]
pub enum Error {
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Internal compiler invariant violated: {0}")]
    InternalInvariant(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("{error} at {span}")]
    SourceAt { error: Box<Error>, span: glrmask_grammar::SourceSpan },

    #[error("Constraint state error: {0}")]
    State(String),
}

impl Error {
    /// Original source coordinates when the frontend can identify them exactly.
    pub fn source_span(&self) -> Option<glrmask_grammar::SourceSpan> {
        match self { Self::SourceAt { span, .. } => Some(*span), _ => None }
    }
    pub(crate) fn with_source_span(self, span: Option<glrmask_grammar::SourceSpan>) -> Self {
        if self.source_span().is_some() { return self; }
        match span { Some(span) => Self::SourceAt { error: Box::new(self), span }, None => self }
    }
}

pub type GlrMaskError = Error;

/// GLRMask operation result; [`Error::source_span`] exposes available source coordinates.
pub type Result<T> = std::result::Result<T, Error>;

impl From<glrmask_grammar::Error> for Error {
    fn from(error: glrmask_grammar::Error) -> Self {
        match error {
            glrmask_grammar::Error::GrammarParse(message) => Self::GrammarParse(message),
            glrmask_grammar::Error::GrammarParseAt{message,span} => Self::GrammarParse(message).with_source_span(Some(span)),
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

impl From<glrmask_vocab::ExactTokenError> for Error {
    fn from(error: glrmask_vocab::ExactTokenError) -> Self {
        Self::Compilation(error.to_string())
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

#[cfg(test)]
mod resource_limit_tests {
    use super::*;

    #[test]
    fn compilation_resource_limit_crosses_rayon_as_a_normal_error() {
        let error = catch_internal_invariant(|| {
            let _: ((), usize) = rayon::join(
                || glrmask_invariant::__private::fail_compilation_resource_limit("projected L1 state budget"),
                || 1,
            );
        }).expect_err("resource-limit signal must cross the Rayon boundary");
        assert!(matches!(error, Error::Compilation(_)));
        assert_eq!(error.to_string(), "Compilation error: resource limit exceeded: projected L1 state budget");
        assert_eq!(catch_internal_invariant(|| 42).unwrap(), 42);
    }
}
