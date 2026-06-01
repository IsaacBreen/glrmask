//! Paper-level scan-relation data vocabulary.
//!
//! For byte fragment `b` scanned from lexer state `q`, the relation records:
//!
//! - the sequence/set of grammar terminals completed wholly inside `b`; and
//! - the lexer state left at the byte boundary.
//!
//! If the boundary is a non-boundary lexer state, a parser-side check is only
//! sound when it is paired with `CanMatch(q')`, the terminals that could still be
//! completed by future bytes.

use crate::grammar::flat::TerminalID;

/// Terminals completed while scanning a byte fragment.
///
/// This wrapper intentionally does not promise uniqueness.  Some callers need a
/// sequence, some need a set, and some keep only the most recent width for a
/// terminal.  The name prevents all of those uses from collapsing back into the
/// vague phrase “possible matches”.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompletedTerminals(pub(crate) Vec<TerminalID>);

/// Lexer state at a fragment boundary when the scan did not fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BoundaryState(pub(crate) u32);

/// Lexer state that is inside a terminal match and therefore requires a
/// continuation check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PartialLexerState(pub(crate) u32);

/// Terminals that can still be completed from a partial lexer state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanMatchSet(pub(crate) Vec<TerminalID>);

/// Result of scanning one byte fragment from one lexer state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ScanOutcome {
    /// The fragment is impossible from the starting state.
    Blocked { completed: CompletedTerminals },
    /// The fragment ends at a lexer boundary; completed terminals are already
    /// complete parser input.
    Complete { completed: CompletedTerminals, boundary: BoundaryState },
    /// The fragment ends inside a terminal match.  The parser must be able to
    /// accept at least one member of `CanMatch(partial)` before the token can be
    /// admitted.
    Partial { completed: CompletedTerminals, partial: PartialLexerState },
}
