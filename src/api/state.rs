//! Public per-generation state type.
//!
//! A [`ConstraintState`] is the mutable state for one generated sequence under a
//! [`Constraint`](crate::api::Constraint).  Mathematically it contains the active
//! frontier of lexer states and parser stacks after the bytes already committed.
//! Operationally it supports:
//!
//! - `mask` / `fill_mask`: evaluate the current frontier against the Parser DWA
//!   and materialize allowed vocabulary tokens.
//! - `commit_token` / `commit_bytes`: scan bytes, advance the parser, and update
//!   the frontier.
//!
//! The shorter [`State`] alias is for internal documentation and examples; the
//! public name remains [`ConstraintState`] for compatibility.

#[doc(inline)]
pub use crate::runtime::ConstraintState;

/// Short alias for [`ConstraintState`].
pub type State<'a> = ConstraintState<'a>;
