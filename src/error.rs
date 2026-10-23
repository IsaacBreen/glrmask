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

/// A parse, build, artifact, or sequence-state failure.
///
/// Source coordinates are included when the frontend can locate them.
#[derive(ThisError, Debug, Clone)]
#[error("{kind}")]
#[non_exhaustive]
pub struct Error {
    kind: ErrorKind,
    span: Option<glrmask_grammar::SourceSpan>,
}

/// The kind of a [`Error`].
#[derive(ThisError, Debug, Clone)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The grammar source could not be parsed.
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),

    /// The grammar could not be compiled.
    #[error("Compilation error: {0}")]
    Compilation(String),

    /// An internal compiler invariant was violated, which indicates a bug.
    #[error("Internal compiler invariant violated: {0}")]
    InternalInvariant(String),

    /// A build exceeded a resource budget.
    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),

    /// The requested strategy cannot represent this grammar exactly.
    #[error("Strategy unsupported: {0}")]
    StrategyUnsupported(String),

    /// An artifact could not be serialized or loaded.
    #[error("Serialization error: {0}")]
    Serialization(String),
}

#[allow(non_snake_case)]
impl Error {
    /// The grammar source could not be parsed.
    pub(crate) fn GrammarParse(message: String) -> Self {
        Self::new(ErrorKind::GrammarParse(message))
    }

    /// The grammar could not be compiled.
    ///
    /// Public only so integration tests and examples can implement
    /// `ParserCompiler`; not part of the supported API surface.
    #[doc(hidden)]
    pub fn Compilation(message: String) -> Self {
        Self::new(ErrorKind::Compilation(message))
    }

    /// An internal compiler invariant was violated, which indicates a bug.
    pub(crate) fn InternalInvariant(message: String) -> Self {
        Self::new(ErrorKind::InternalInvariant(message))
    }

    /// A build exceeded a resource budget.
    pub(crate) fn ResourceLimit(message: String) -> Self {
        Self::new(ErrorKind::ResourceLimit(message))
    }

    /// The requested strategy cannot represent this grammar exactly.
    pub(crate) fn StrategyUnsupported(message: String) -> Self {
        Self::new(ErrorKind::StrategyUnsupported(message))
    }

    /// An artifact could not be serialized or loaded.
    pub(crate) fn Serialization(message: String) -> Self {
        Self::new(ErrorKind::Serialization(message))
    }

    fn new(kind: ErrorKind) -> Self {
        Self { kind, span: None }
    }

    /// The failure kind.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Original source coordinates when the frontend can identify them exactly.
    pub fn source_span(&self) -> Option<glrmask_grammar::SourceSpan> {
        self.span
    }

    pub(crate) fn with_source_span(mut self, span: Option<glrmask_grammar::SourceSpan>) -> Self {
        if self.span.is_none() {
            self.span = span;
        }
        self
    }

    /// A template strategy declined deterministically. Public entry points
    /// return this error unchanged, or attempt the opt-in terminal-ended
    /// rescue when it is explicitly enabled.
    pub(crate) fn is_strategy_decline(&self) -> bool {
        match &self.kind {
            ErrorKind::ResourceLimit(message) if message.starts_with("boundary_parser_dwa:") => false,
            ErrorKind::ResourceLimit(_) | ErrorKind::StrategyUnsupported(_) => true,
            _ => false,
        }
    }

    pub(crate) fn decline_stage(&self) -> &'static str {
        match &self.kind {
            ErrorKind::ResourceLimit(message) if message.starts_with("glr_table_lr1") => "glr_table_lr1",
            ErrorKind::ResourceLimit(message) if message.starts_with("glr_table_collapse") => "glr_table_collapse",
            ErrorKind::ResourceLimit(message) if message.starts_with("glr_table") => "glr_table",
            ErrorKind::ResourceLimit(message) if message.starts_with("o2_quotient_materialize") => "o2_quotient_materialize",
            ErrorKind::ResourceLimit(message) if message.starts_with("runtime_tokenizer_join") => "runtime_tokenizer_join",
            ErrorKind::ResourceLimit(message) if message.starts_with("split_terminal_dwa_wall") => "split_terminal_dwa_wall",
            ErrorKind::ResourceLimit(message) if message.starts_with("template_determinization") => "template_determinization",
            ErrorKind::ResourceLimit(message) if message.starts_with("template_specialization") => "template_specialization",
            ErrorKind::ResourceLimit(message) if message.starts_with("grammar_normalization") => "grammar_normalization",
            ErrorKind::ResourceLimit(message) if message.starts_with("single_use_inlining") => "single_use_inlining",
            ErrorKind::ResourceLimit(message) if message.starts_with("bound_reduction_length") => "bound_reduction_length",
            ErrorKind::ResourceLimit(message) if message.starts_with("boundary_parser_dwa:") => "boundary_parser_dwa",
            ErrorKind::ResourceLimit(message) if message.starts_with("parser_dwa") => "parser_dwa",
            ErrorKind::ResourceLimit(_) => "template",
            ErrorKind::StrategyUnsupported(message) if message.contains("right-recursive cycle") => "glr_table",
            ErrorKind::StrategyUnsupported(_) => "template",
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

/// A GLRMask operation result.
///
/// [`Error::source_span`] exposes available source coordinates.
pub type Result<T> = std::result::Result<T, Error>;

impl From<glrmask_grammar::Error> for Error {
    fn from(error: glrmask_grammar::Error) -> Self {
        match error {
            glrmask_grammar::Error::GrammarParse(message) => Self::GrammarParse(message),
            glrmask_grammar::Error::GrammarParseAt { message, span } => {
                Self::GrammarParse(message).with_source_span(Some(span))
            }
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

        assert!(matches!(error.kind(), ErrorKind::InternalInvariant(_)));
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
        assert!(matches!(error.kind(), ErrorKind::ResourceLimit(_)));
        assert_eq!(error.to_string(), "Resource limit exceeded: projected L1 state budget");
        assert_eq!(catch_internal_invariant(|| 42).unwrap(), 42);
    }
}
