use thiserror::Error as ThisError;

pub(crate) fn catch_internal_invariant<T>(f: impl FnOnce() -> T) -> Result<T> {
    glrmask_invariant::__private::catch_compilation_resource_limit(|| {
        glrmask_invariant::__private::catch_internal_invariant_message(f)
    })
    .map_err(Error::ResourceLimit)?
    .map_err(Error::InternalInvariant)
}

/// Run a fallible template attempt while letting a deterministic resource-limit
/// unwind become a typed decline. Public entry points use this so a declined
/// template strategy can return that decline, or attempt the opt-in
/// terminal-ended rescue when it is explicitly enabled.
pub(crate) fn catch_template_attempt<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_internal_invariant(f).and_then(|result| result)
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

    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),

    /// The requested template strategy cannot represent this grammar exactly.
    /// Public entry points return this error unchanged unless the opt-in
    /// terminal-ended rescue is enabled; if the rescue also declines, this
    /// error is returned to the caller unchanged.
    #[error("Strategy unsupported: {0}")]
    StrategyUnsupported(String),

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

    /// A template strategy declined deterministically. Public entry points
    /// return this error unchanged, or attempt the opt-in terminal-ended
    /// rescue when it is explicitly enabled.
    pub(crate) fn is_strategy_decline(&self) -> bool {
        match self {
            Self::ResourceLimit(_) | Self::StrategyUnsupported(_) => true,
            Self::SourceAt { error, .. } => error.is_strategy_decline(),
            _ => false,
        }
    }

    pub(crate) fn decline_stage(&self) -> &'static str {
        match self {
            Self::ResourceLimit(message) if message.starts_with("glr_table_lr1") => "glr_table_lr1",
            Self::ResourceLimit(message) if message.starts_with("glr_table_collapse") => "glr_table_collapse",
            Self::ResourceLimit(message) if message.starts_with("glr_table") => "glr_table",
            Self::ResourceLimit(message) if message.starts_with("o2_quotient_materialize") => "o2_quotient_materialize",
            Self::ResourceLimit(message) if message.starts_with("runtime_tokenizer_join") => "runtime_tokenizer_join",
            Self::ResourceLimit(message) if message.starts_with("split_terminal_dwa_wall") => "split_terminal_dwa_wall",
            Self::ResourceLimit(message) if message.starts_with("template_determinization") => "template_determinization",
            Self::ResourceLimit(message) if message.starts_with("template_specialization") => "template_specialization",
            Self::ResourceLimit(message) if message.starts_with("grammar_normalization") => "grammar_normalization",
            Self::ResourceLimit(message) if message.starts_with("single_use_inlining") => "single_use_inlining",
            Self::ResourceLimit(message) if message.starts_with("bound_reduction_length") => "bound_reduction_length",
            Self::ResourceLimit(message) if message.starts_with("parser_dwa") => "parser_dwa",
            Self::ResourceLimit(_) => "template",
            Self::StrategyUnsupported(message) if message.contains("right-recursive cycle") => "glr_table",
            Self::StrategyUnsupported(_) => "template",
            Self::SourceAt { error, .. } => error.decline_stage(),
            _ => "template",
        }
    }

    pub(crate) fn classify_template_compile_message(message: String) -> Self {
        if message.contains("right-recursive cycle") || message.contains("reduction cycle") {
            Self::StrategyUnsupported(message)
        } else {
            Self::Compilation(message)
        }
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
        assert!(matches!(error, Error::ResourceLimit(_)));
        assert_eq!(error.to_string(), "Resource limit exceeded: projected L1 state budget");
        assert_eq!(catch_internal_invariant(|| 42).unwrap(), 42);
    }
}
