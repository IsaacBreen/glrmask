/// The one typed decline that is a boundary request, not a mask strategy
/// decline: `InternalBoundaryMaskStrategy::ParserDwa` refused the boundary parser DWA.
pub(crate) const BOUNDARY_DWA_DECLINE: &str =
    "the parser DWA for a grammar boundary would be too large; use the Auto or FastCompile boundary mask strategy";

// Only deterministic plan-specific refusals qualify for the public soft
// preferences. A general ResourceLimit (allocation failure, graph size,
// parser control closure, etc.) is not a reason to conceal a real failure.
pub(crate) fn is_soft_strategy_decline(error: &Error) -> bool {
    match error.kind() {
        ErrorKind::StrategyUnsupported => true,
        ErrorKind::ResourceLimit => {
            let m = error.message();
            m.ends_with(" build budget exhausted")
                || m == "the grammar is too large to normalize"
                || (m.starts_with("projected L1 partition ") && m.contains("exceeding GLRMASK_L1_SINGLE_MAX_STATES="))
        }
        _ => false,
    }
}

pub(crate) fn catch_internal_invariant<T>(f: impl FnOnce() -> T) -> Result<T> {
    glrmask_invariant::__private::catch_compilation_resource_limit(|| {
        glrmask_invariant::__private::catch_internal_invariant_message(f)
    })
    .map_err(Error::ResourceLimit)?
    .map_err(Error::Internal)
}

/// Run a fallible template attempt while letting a deterministic resource-limit
/// unwind become a typed decline. Public entry points use this so a declined
/// template strategy can return that decline, or attempt the opt-in
/// terminal-ended rescue when it is explicitly enabled.
pub(crate) fn catch_template_attempt<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_internal_invariant(f).and_then(|result| result)
}

pub use glrmask_vocab::{Error, ErrorKind};
use glrmask_vocab::__private as error_internal;

/// Crate-internal constructors and helpers for [`Error`]. The type lives in
/// glrmask-vocab (so `Vocab::new` can return it), so these are trait methods;
/// bring the trait into scope to write `Error::Compilation(message)`.
#[allow(non_snake_case)]
pub(crate) trait ErrorExt: Sized {
    /// The grammar source or JSON Schema could not be parsed.
    fn Syntax(message: String) -> Self;
    /// The caller passed something invalid.
    fn InvalidInput(message: String) -> Self;
    /// A valid grammar could not be compiled.
    fn Compilation(message: String) -> Self;
    /// A bug in GLRMask: an internal invariant was violated.
    fn Internal(message: String) -> Self;
    /// A build exceeded a resource budget.
    fn ResourceLimit(message: String) -> Self;
    /// The requested strategy cannot represent this grammar exactly.
    fn StrategyUnsupported(message: String) -> Self;
    /// An artifact could not be serialized or loaded.
    fn Serialization(message: String) -> Self;
    fn VocabMismatch(message: String) -> Self;
    /// The message, without the kind prefix or source span.
    fn message(&self) -> &str;
    /// Sets the source span if it isn't set yet.
    fn with_source_span(self, span: Option<glrmask_grammar::SourceSpan>) -> Self;
    /// A template strategy declined deterministically. Public entry points
    /// return this error unchanged, or attempt the opt-in terminal-ended
    /// rescue when it is explicitly enabled.
    fn is_strategy_decline(&self) -> bool;
    fn decline_stage(&self) -> &'static str;
    fn classify_template_compile_message(message: String) -> Self;
}

impl ErrorExt for Error {
    fn Syntax(message: String) -> Self {
        error_internal::new_error(ErrorKind::Syntax, message)
    }
    fn InvalidInput(message: String) -> Self {
        error_internal::new_error(ErrorKind::InvalidInput, message)
    }
    fn Compilation(message: String) -> Self {
        error_internal::new_error(ErrorKind::Compilation, message)
    }
    fn Internal(message: String) -> Self {
        error_internal::new_error(ErrorKind::Internal, message)
    }
    fn ResourceLimit(message: String) -> Self {
        error_internal::new_error(ErrorKind::ResourceLimit, message)
    }
    fn StrategyUnsupported(message: String) -> Self {
        error_internal::new_error(ErrorKind::StrategyUnsupported, message)
    }
    fn Serialization(message: String) -> Self {
        error_internal::new_error(ErrorKind::Serialization, message)
    }
    fn VocabMismatch(message: String) -> Self {
        error_internal::new_error(ErrorKind::VocabMismatch, message)
    }
    fn message(&self) -> &str {
        error_internal::error_message(self)
    }
    fn with_source_span(self, span: Option<glrmask_grammar::SourceSpan>) -> Self {
        error_internal::with_source_span(self, span)
    }
    fn is_strategy_decline(&self) -> bool {
        match self.kind() {
            ErrorKind::ResourceLimit => self.message() != BOUNDARY_DWA_DECLINE,
            ErrorKind::StrategyUnsupported => true,
            _ => false,
        }
    }
    fn decline_stage(&self) -> &'static str {
        // Order matters: a longer stage name before its prefix.
        const STAGES: [&str; 11] = [
            "glr_table_lr1",
            "glr_table_collapse",
            "glr_table",
            "o2_quotient_materialize",
            "runtime_tokenizer_join",
            "split_terminal_dwa_wall",
            "template_determinization",
            "template_specialization",
            "grammar_normalization",
            "single_use_inlining",
            "bound_reduction_length",
        ];
        let message = self.message();
        match self.kind() {
            ErrorKind::ResourceLimit => {
                if let Some(stage) = STAGES.into_iter().find(|stage| message.starts_with(stage)) {
                    stage
                } else if message == BOUNDARY_DWA_DECLINE {
                    "boundary_parser_dwa"
                } else if message.starts_with("parser_dwa") {
                    "parser_dwa"
                } else {
                    "template"
                }
            }
            ErrorKind::StrategyUnsupported if message.contains("right-recursive cycle") => "glr_table",
            _ => "template",
        }
    }
    fn classify_template_compile_message(message: String) -> Self {
        if message.contains("right-recursive cycle") || message.contains("reduction cycle") {
            Self::StrategyUnsupported(message)
        } else {
            Self::Compilation(message)
        }
    }
}

/// Converts a grammar front-end error. Not a `From` impl, so the internal
/// crate's error type never becomes part of the public API.
pub(crate) fn from_grammar(error: glrmask_grammar::Error) -> Error {
    match error {
        glrmask_grammar::Error::GrammarParse(message) => Error::Syntax(message),
        glrmask_grammar::Error::GrammarParseAt { message, span } => {
            Error::Syntax(message).with_source_span(Some(span))
        }
    }
}

/// Converts a weighted-automata error.
pub(crate) fn from_automata(error: glrmask_weighted_automata::Error) -> Error {
    match error {
        glrmask_weighted_automata::Error::Compilation(message) => Error::Compilation(message),
    }
}

/// A `Compilation` error, for integration tests and examples that implement
/// `ParserCompiler`. Not part of the supported API.
#[cfg(feature = "internal-api")]
pub fn compilation_error(message: String) -> Error {
    Error::Compilation(message)
}

/// Rejects a token ID above the maximum supported token ID.
pub(crate) fn check_token_id(id: u32) -> Result<()> {
    if id > glrmask_vocab::__private::MAX_TOKEN_ID {
        return Err(Error::InvalidInput(format!(
            "token ID {id} is outside the supported range"
        )));
    }
    Ok(())
}

/// Rejects a token set with an ID above the maximum supported token ID.
pub(crate) fn check_token_set(tokens: &glrmask_vocab::TokenSet) -> Result<()> {
    match tokens.range_pairs().last() {
        Some(&(_, end)) => check_token_id(end),
        None => Ok(()),
    }
}

/// Rejects a vocabulary with an ID above the maximum supported token ID,
/// and checks that this machine can allocate the per-token-ID tables a compile
/// builds, so a lack of memory is an error rather than an abort.
pub(crate) fn check_vocab_token_ids(vocab: &glrmask_vocab::Vocab) -> Result<()> {
    let Some(max) = vocab.max_event_token_id() else {
        return Ok(());
    };
    check_token_id(max)?;
    let len = max as usize + 1;
    Vec::<u32>::new().try_reserve_exact(len).map_err(|_| token_table_error(len))
}

fn token_table_error(len: usize) -> Error {
    Error::ResourceLimit(format!("not enough memory for a table of {len} token IDs"))
}

/// A vector of `len` copies of `value`, sized by a token ID. A failed
/// allocation is a `ResourceLimit` error instead of an abort.
pub(crate) fn try_token_table<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut table = Vec::new();
    table.try_reserve_exact(len).map_err(|_| token_table_error(len))?;
    table.resize(len, value);
    Ok(table)
}

/// A result with a GLRMask [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

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

        assert_eq!(error.kind(), ErrorKind::Internal);
        assert_eq!(
            error.to_string(),
            "internal error (a bug in GLRMask; please report it): analysis coordinate escaped its domain"
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
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
        assert_eq!(error.to_string(), "resource limit exceeded: projected L1 state budget");
        assert_eq!(catch_internal_invariant(|| 42).unwrap(), 42);
    }
}

#[cfg(test)]
#[test]
fn soft_preference_fallback_distinguishes_resource_failures() {
    assert!(is_soft_strategy_decline(&Error::ResourceLimit("the grammar is too large to normalize".into())));
    assert!(!is_soft_strategy_decline(&Error::ResourceLimit("not enough memory for a table of 128 token IDs".into())));
    assert!(!is_soft_strategy_decline(&Error::InvalidInput("bad input".into())));
}
