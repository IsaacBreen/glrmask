//! Reserved public option types.
//!
//! The current constructors use the crate defaults.  This module establishes the
//! publication-facing names for future option-bearing constructors without
//! forcing environment-variable details into the public API.  Environment parsing
//! belongs under `crate::diagnostics` / future `crate::config`, not in user code.

/// Compile-time options for future option-bearing constructors.
///
/// Today this is a zero-sized marker for the default compilation semantics.  It
/// is non-exhaustive so fields can be added without a breaking change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompileOptions {}

/// Runtime options for future state/mask/commit customization.
///
/// Today this is a zero-sized marker for the default runtime semantics.  It is
/// non-exhaustive so fields can be added without a breaking change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RuntimeOptions {}
