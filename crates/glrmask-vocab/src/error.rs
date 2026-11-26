//! GLRMask's public error type. It lives in this crate so that `Vocab::new` can
//! return it; the `glrmask` crate re-exports it as `glrmask::Error`.
use std::fmt;

/// A location in grammar source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SourceSpan {
    /// Byte offset of the first byte.
    pub start: usize,
    /// Byte offset just past the last byte.
    pub end: usize,
    /// Line of `start`, counting from 1.
    pub line: usize,
    /// Column of `start`, counting characters from 1.
    pub column: usize,
}

impl SourceSpan {
    #[doc(hidden)]
    pub const fn new(start: usize, end: usize, line: usize, column: usize) -> Self {
        Self { start, end, line, column }
    }
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

#[derive(Debug, Clone)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    span: Option<SourceSpan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The grammar source (GLRM or Lark) or JSON Schema can't be parsed.
    Syntax,
    /// An invalid argument, binding or token ID.
    InvalidInput,
    /// A valid grammar that GLRMask can't compile.
    Compilation,
    /// A build exceeded a size or time budget.
    ResourceLimit,
    /// The chosen mask strategy can't handle this grammar; another strategy may work.
    StrategyUnsupported,
    /// Saved bytes are corrupt, from another GLRMask version, or for another vocabulary.
    Serialization,
    /// A bug in GLRMask.
    Internal,
}

impl Error {
    /// Currently one of `ErrorKind::Syntax`, `ErrorKind::InvalidInput`, `ErrorKind::Compilation`, `ErrorKind::ResourceLimit`, `ErrorKind::StrategyUnsupported`, `ErrorKind::Serialization`, or `ErrorKind::Internal`.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Where in the grammar source the error is, if known.
    pub fn source_span(&self) -> Option<SourceSpan> {
        self.span
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ErrorKind::Syntax => write!(f, "grammar error: {}", self.message)?,
            ErrorKind::ResourceLimit => write!(f, "resource limit exceeded: {}", self.message)?,
            ErrorKind::StrategyUnsupported => {
                write!(f, "unsupported by the chosen mask strategy: {}", self.message)?
            }
            ErrorKind::Serialization => write!(f, "cannot load saved bytes: {}", self.message)?,
            ErrorKind::Internal => write!(
                f,
                "internal error (a bug in GLRMask; please report it): {}",
                self.message
            )?,
            ErrorKind::InvalidInput | ErrorKind::Compilation => f.write_str(&self.message)?,
        }
        if let Some(span) = self.span {
            write!(f, " ({span})")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

/// Constructors and accessors for the rest of the workspace; not a stable API.
pub(crate) mod internal {
    use super::*;

    /// A new error of `kind` with `message`.
    pub fn new_error(kind: ErrorKind, message: String) -> Error {
        Error { kind, message, span: None }
    }

    /// The error's message, without the kind prefix or source span.
    pub fn error_message(error: &Error) -> &str {
        &error.message
    }

    /// Sets the source span if it isn't set yet.
    pub fn with_source_span(mut error: Error, span: Option<SourceSpan>) -> Error {
        if error.span.is_none() {
            error.span = span;
        }
        error
    }
}
