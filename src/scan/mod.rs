//! Shared scan-domain vocabulary.
//!
//! The implementation has two users of lexer scanning:
//!
//! 1. compile-time scan-relation construction, which asks what terminals can be
//!    completed by a vocabulary token from every lexer state; and
//! 2. runtime commit, which scans the bytes already chosen by the model and
//!    advances the parser through completed terminal boundaries.
//!
//! This module gives both users the same language without forcing them to share
//! stateful implementation details.  It is deliberately lower-level than
//! `compile::scan_relation`: it says what scanning *means*, not how to compress
//! all scan outcomes into runtime weights.

pub(crate) mod execution;
pub(crate) mod relation;

pub(crate) use relation::{
    BoundaryState,
    CanMatchSet,
    CompletedTerminals,
    PartialLexerState,
    ScanOutcome,
};
